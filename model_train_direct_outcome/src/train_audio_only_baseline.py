import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from baseline_utils import MURMUR_LABELS, OUTCOME_LABELS, search_recall_threshold, write_diagnostics
from dataset_direct_outcome import CirCorDirectOutcomeDataset, direct_outcome_collate
from dataset_hms import POSITIONS
from model_hms import ScaleEncoder
from paths import CHECKPOINT_ROOT, RESULTS_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


class AudioOnlyOutcomeModel(nn.Module):
    def __init__(
        self,
        branch_dim=64,
        segment_embedding_dim=128,
        patient_embedding_dim=160,
        num_positions=4,
        position_dim=16,
    ):
        super().__init__()
        self.scale1_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale2_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale3_encoder = ScaleEncoder(out_dim=branch_dim)
        self.position_embedding = nn.Embedding(num_positions, position_dim)
        fused_dim = branch_dim * 3 + position_dim
        self.segment_head = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, segment_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Linear(segment_embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.patient_head = nn.Sequential(
            nn.Linear(segment_embedding_dim * 2, patient_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(patient_embedding_dim, patient_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.outcome_head = nn.Linear(patient_embedding_dim, 2)

    def _encode_segments(self, x_scale1, x_scale2, x_scale3, position_index):
        batch_size, num_segments = x_scale1.shape[:2]
        flat_x1 = x_scale1.reshape(batch_size * num_segments, *x_scale1.shape[2:])
        flat_x2 = x_scale2.reshape(batch_size * num_segments, *x_scale2.shape[2:])
        flat_x3 = x_scale3.reshape(batch_size * num_segments, *x_scale3.shape[2:])
        flat_positions = position_index.reshape(batch_size * num_segments)

        f1 = self.scale1_encoder(flat_x1)
        f2 = self.scale2_encoder(flat_x2)
        f3 = self.scale3_encoder(flat_x3)
        pos = self.position_embedding(flat_positions.long())
        segment_features = torch.cat([f1, f2, f3, pos], dim=1)
        segment_embeddings = self.segment_head(segment_features)
        return segment_embeddings.view(batch_size, num_segments, -1)

    def forward(self, x_scale1, x_scale2, x_scale3, position_index, segment_mask, return_embedding=False):
        segment_embeddings = self._encode_segments(x_scale1, x_scale2, x_scale3, position_index)
        mask = segment_mask.float().clamp(0.0, 1.0)
        attention_logits = self.attention(segment_embeddings).squeeze(-1)
        attention_logits = attention_logits.masked_fill(mask <= 0.0, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1) * mask
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attentive_pool = torch.sum(segment_embeddings * attention_weights.unsqueeze(-1), dim=1)

        max_pool = segment_embeddings.masked_fill(mask.unsqueeze(-1) <= 0.0, -1e4).max(dim=1).values
        max_pool = torch.where(max_pool < -1e3, torch.zeros_like(max_pool), max_pool)
        patient_embedding = self.patient_head(torch.cat([attentive_pool, max_pool], dim=1))
        logits = self.outcome_head(patient_embedding)
        if return_embedding:
            return {
                "outcome_logits": logits,
                "patient_embedding": patient_embedding,
                "attention_weights": attention_weights,
            }
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-only outcome baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--outcome-abnormal-weight", type=float, default=1.0)
    parser.add_argument("--max-segments-per-patient", type=int, default=32)
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "audio_only_baseline.pth"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_ROOT / "audio_only_baseline"))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["x_scale1"],
            batch["x_scale2"],
            batch["x_scale3"],
            batch["position_index"],
            batch["segment_mask"],
        )
        loss = F.cross_entropy(logits, batch["y_outcome"], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def collect_predictions(model, loader, dataset, device):
    raw_df = dataset.df.set_index("Patient ID", drop=False)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            outputs = model(
                moved["x_scale1"],
                moved["x_scale2"],
                moved["x_scale3"],
                moved["position_index"],
                moved["segment_mask"],
                return_embedding=True,
            )
            probs = torch.softmax(outputs["outcome_logits"], dim=1)[:, 1].detach().cpu().numpy()
            attention = outputs["attention_weights"].detach().cpu().numpy()
            position_index = batch["position_index"].cpu().numpy()
            segment_mask = batch["segment_mask"].cpu().numpy()
            for i, patient_id in enumerate(batch["patient_id"]):
                raw = raw_df.loc[str(patient_id)] if str(patient_id) in raw_df.index else {}
                y_outcome = int(batch["y_outcome"][i].item())
                y_murmur = int(batch["y_murmur"][i].item())
                valid = segment_mask[i] > 0.0
                pos_values = position_index[i][valid]
                weights = attention[i][valid] if valid.any() else np.asarray([])
                attention_by_position = {f"attention_{pos}": 0.0 for pos in POSITIONS}
                top_attention_position = ""
                top_attention_weight = np.nan
                if len(weights) > 0:
                    top_idx = int(np.argmax(weights))
                    top_attention_position = POSITIONS[int(pos_values[top_idx])]
                    top_attention_weight = float(weights[top_idx])
                    for pos_idx, pos_name in enumerate(POSITIONS):
                        attention_by_position[f"attention_{pos_name}"] = float(weights[pos_values == pos_idx].sum())

                row = {
                    "patient_id": str(patient_id),
                    "prob_abnormal": float(probs[i]),
                    "y_outcome": y_outcome,
                    "outcome": OUTCOME_LABELS.get(y_outcome, str(y_outcome)),
                    "y_murmur": y_murmur,
                    "murmur": MURMUR_LABELS.get(y_murmur, str(y_murmur)),
                    "position_count": int(batch["position_count"][i].item()),
                    "has_all_positions": bool(batch["has_all_positions"][i].item()),
                    "available_positions": ",".join(batch["available_positions"][i]),
                    "segment_count": int(valid.sum()),
                    "top_attention_position": top_attention_position,
                    "top_attention_weight": top_attention_weight,
                    "Age": raw.get("Age", ""),
                    "Sex": raw.get("Sex", ""),
                    "Height": raw.get("Height", ""),
                    "Weight": raw.get("Weight", ""),
                    "Pregnancy status": raw.get("Pregnancy status", ""),
                }
                for pos_name in POSITIONS:
                    row[f"has_{pos_name}"] = int(pos_name in batch["available_positions"][i])
                    row[f"segments_{pos_name}"] = int(np.sum(pos_values == POSITIONS.index(pos_name)))
                row.update(attention_by_position)
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate(model, loader, dataset, device, target_recall):
    df = collect_predictions(model, loader, dataset, device)
    return search_recall_threshold(
        df["y_outcome"].to_numpy(dtype=np.int64),
        df["prob_abnormal"].to_numpy(dtype=np.float32),
        target_recall,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print("Using device:", device)
    dataset = CirCorDirectOutcomeDataset(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        max_segments_per_patient=args.max_segments_per_patient,
    )
    indices = list(range(len(dataset)))
    labels = [dataset.patients[index]["y_outcome"] for index in indices]
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    print("Train outcome counts:", dict(Counter(dataset.patients[idx]["y_outcome"] for idx in train_indices)))
    print("Val outcome counts:", dict(Counter(dataset.patients[idx]["y_outcome"] for idx in val_indices)))

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    )
    model = AudioOnlyOutcomeModel().to(device)
    class_weights = torch.tensor([1.0, args.outcome_abnormal_weight], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = None
    best_state = None
    best_metric = None
    patience = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        metric = evaluate(model, val_loader, dataset, device, args.target_recall)
        score = (
            metric["abnormal_recall"] >= args.target_recall,
            metric["specificity"],
            metric["accuracy"],
            -metric["fn"],
            -metric["fp"],
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f} | threshold={metric['threshold']:.2f} | "
            f"acc={metric['accuracy']:.4f} | recall={metric['abnormal_recall']:.4f} | "
            f"specificity={metric['specificity']:.4f} | FN={metric['fn']} | FP={metric['fp']}"
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            print("New best audio-only baseline checkpoint.")
        else:
            patience += 1
            print(f"No improvement for {patience} epoch(s).")
        if patience >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "args": vars(args),
            "best_metric": best_metric,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    pred_df = collect_predictions(model, val_loader, dataset, device)
    summary = write_diagnostics(
        pred_df,
        args.output_dir,
        target_recall=args.target_recall,
        extra_summary={
            "model_type": "audio_only_baseline",
            "checkpoint_path": str(checkpoint_path),
            "feature_note": "Audio log-mel inputs and segment position embedding only; no clinical branch or explicit missingness flags.",
        },
    )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved checkpoint:", checkpoint_path)
    print("Saved metadata:", metadata_path)
    print("Wrote diagnostics:", args.output_dir)


if __name__ == "__main__":
    main()
