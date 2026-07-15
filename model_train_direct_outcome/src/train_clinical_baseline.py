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
from torch.utils.data import DataLoader, Dataset, Subset

from baseline_utils import MURMUR_LABELS, OUTCOME_LABELS, metrics_from_probs, search_recall_threshold, write_diagnostics
from dataset_direct_outcome import AGE_MAP, SEX_MAP
from dataset_hms import POSITIONS
from paths import CHECKPOINT_ROOT, RESULTS_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


class ClinicalBaselineDataset(Dataset):
    def __init__(self, csv_path, wav_dir):
        self.csv_path = Path(csv_path)
        self.wav_dir = Path(wav_dir)
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [column.strip() for column in self.df.columns]
        self.df["Patient ID"] = self.df["Patient ID"].astype(str)
        self.df = self.df[self.df["Murmur"].isin(["Absent", "Present", "Unknown"])].copy()
        self.df = self.df[self.df["Outcome"].isin(["Normal", "Abnormal"])].copy()
        self.murmur_map = {"Absent": 0, "Present": 1, "Unknown": 2}
        self.outcome_map = {"Normal": 0, "Abnormal": 1}

        self.patients = []
        for _, row in self.df.iterrows():
            patient_id = str(row["Patient ID"])
            available_positions = [
                pos for pos in POSITIONS if (self.wav_dir / f"{patient_id}_{pos}.wav").exists()
            ]
            if not available_positions:
                continue
            features = self._features(row, available_positions)
            self.patients.append(
                {
                    "patient_id": patient_id,
                    "features": features,
                    "y_outcome": int(self.outcome_map[row["Outcome"]]),
                    "y_murmur": int(self.murmur_map[row["Murmur"]]),
                    "available_positions": available_positions,
                    "position_count": len(available_positions),
                    "has_all_positions": len(available_positions) == len(POSITIONS),
                    "raw": row.to_dict(),
                }
            )
        print(f"Loaded {len(self.patients)} patients with at least one wav")

    @property
    def feature_dim(self):
        return len(self.patients[0]["features"])

    def _features(self, row, available_positions):
        age_raw = row.get("Age", np.nan)
        age_missing = float(pd.isna(age_raw) or str(age_raw).strip().lower() == "nan")
        age_value = AGE_MAP.get(str(age_raw).strip(), 2.0) / 4.0

        sex_raw = row.get("Sex", np.nan)
        sex_missing = float(pd.isna(sex_raw) or str(sex_raw).strip().lower() == "nan")
        sex_value = SEX_MAP.get(str(sex_raw).strip(), 0.5)

        height_raw = pd.to_numeric(row.get("Height", np.nan), errors="coerce")
        height_missing = float(pd.isna(height_raw))
        height_value = 0.0 if pd.isna(height_raw) else float(np.clip(height_raw / 200.0, 0.0, 1.5))

        weight_raw = pd.to_numeric(row.get("Weight", np.nan), errors="coerce")
        weight_missing = float(pd.isna(weight_raw))
        weight_value = 0.0 if pd.isna(weight_raw) else float(np.clip(weight_raw / 120.0, 0.0, 1.5))

        pregnancy_raw = row.get("Pregnancy status", np.nan)
        pregnancy_missing = float(pd.isna(pregnancy_raw) or str(pregnancy_raw).strip().lower() == "nan")
        pregnancy_text = str(pregnancy_raw).strip().lower()
        pregnancy_value = 1.0 if pregnancy_text == "true" else 0.0 if pregnancy_text == "false" else 0.5

        position_flags = [float(pos in available_positions) for pos in POSITIONS]
        position_count = float(len(available_positions) / len(POSITIONS))
        has_all_positions = float(len(available_positions) == len(POSITIONS))

        return np.asarray(
            [
                age_value,
                sex_value,
                height_value,
                weight_value,
                pregnancy_value,
                age_missing,
                sex_missing,
                height_missing,
                weight_missing,
                pregnancy_missing,
                position_count,
                has_all_positions,
                *position_flags,
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        return {
            "patient_id": patient["patient_id"],
            "features": torch.tensor(patient["features"], dtype=torch.float32),
            "y_outcome": torch.tensor(patient["y_outcome"], dtype=torch.long),
            "y_murmur": torch.tensor(patient["y_murmur"], dtype=torch.long),
            "position_count": torch.tensor(patient["position_count"], dtype=torch.long),
            "has_all_positions": torch.tensor(patient["has_all_positions"], dtype=torch.bool),
            "available_positions": patient["available_positions"],
            "raw": patient["raw"],
        }


def collate(batch):
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "y_outcome": torch.stack([item["y_outcome"] for item in batch], dim=0),
        "y_murmur": torch.stack([item["y_murmur"] for item in batch], dim=0),
        "position_count": torch.stack([item["position_count"] for item in batch], dim=0),
        "has_all_positions": torch.stack([item["has_all_positions"] for item in batch], dim=0),
        "available_positions": [item["available_positions"] for item in batch],
        "raw": [item["raw"] for item in batch],
    }


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features):
        return self.net(features)


def parse_args():
    parser = argparse.ArgumentParser(description="Train clinical-only shortcut baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--outcome-abnormal-weight", type=float, default=1.0)
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "clinical_only_baseline.pth"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_ROOT / "clinical_only_baseline"))
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
        logits = model(batch["features"])
        loss = F.cross_entropy(logits, batch["y_outcome"], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def collect_predictions(model, loader, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            logits = model(moved["features"])
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for i, patient_id in enumerate(batch["patient_id"]):
                raw = batch["raw"][i]
                y_outcome = int(batch["y_outcome"][i].item())
                y_murmur = int(batch["y_murmur"][i].item())
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "prob_abnormal": float(probs[i]),
                        "y_outcome": y_outcome,
                        "outcome": OUTCOME_LABELS.get(y_outcome, str(y_outcome)),
                        "y_murmur": y_murmur,
                        "murmur": MURMUR_LABELS.get(y_murmur, str(y_murmur)),
                        "position_count": int(batch["position_count"][i].item()),
                        "has_all_positions": bool(batch["has_all_positions"][i].item()),
                        "available_positions": ",".join(batch["available_positions"][i]),
                        "Age": raw.get("Age", ""),
                        "Sex": raw.get("Sex", ""),
                        "Height": raw.get("Height", ""),
                        "Weight": raw.get("Weight", ""),
                        "Pregnancy status": raw.get("Pregnancy status", ""),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(model, loader, device, target_recall):
    df = collect_predictions(model, loader, device)
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
    dataset = ClinicalBaselineDataset(args.csv_path, args.wav_dir)
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
    print("Feature dim:", dataset.feature_dim)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    model = ClinicalMLP(dataset.feature_dim).to(device)
    class_weights = torch.tensor([1.0, args.outcome_abnormal_weight], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = None
    best_state = None
    best_metric = None
    patience = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        metric = evaluate(model, val_loader, device, args.target_recall)
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
            print("New best clinical-only baseline checkpoint.")
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
            "feature_dim": dataset.feature_dim,
            "args": vars(args),
            "best_metric": best_metric,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    pred_df = collect_predictions(model, val_loader, device)
    summary = write_diagnostics(
        pred_df,
        args.output_dir,
        target_recall=args.target_recall,
        extra_summary={
            "model_type": "clinical_only_baseline",
            "checkpoint_path": str(checkpoint_path),
            "feature_dim": dataset.feature_dim,
            "feature_note": "Clinical values, clinical missing flags, position_count, has_all_positions, and per-position presence flags.",
        },
    )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved checkpoint:", checkpoint_path)
    print("Saved metadata:", metadata_path)
    print("Wrote diagnostics:", args.output_dir)


if __name__ == "__main__":
    main()
