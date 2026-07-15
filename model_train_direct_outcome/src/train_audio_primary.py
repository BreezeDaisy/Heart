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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from baseline_utils import MURMUR_LABELS, OUTCOME_LABELS, search_recall_threshold, write_diagnostics
from dataset_audio_primary import CirCorAudioPrimaryDataset, audio_primary_collate
from dataset_hms import POSITIONS
from model_hms import ScaleEncoder
from paths import CHECKPOINT_ROOT, RESULTS_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


class AudioPrimaryOutcomeModel(nn.Module):
    """Audio-first patient-level outcome model.

    Inputs are intentionally limited to:
    - multi-scale log-mel heart sound segments
    - segment auscultation position embedding
    - compact handcrafted acoustic descriptors
    - weak clinical values: age, sex, height, weight

    It does not receive position_count, has_all_positions, missingness flags,
    pregnancy status, or Murmur. Those fields remain available only in
    diagnostics so we can check whether the model still behaves differently
    across subgroups.
    """

    def __init__(
        self,
        branch_dim=64,
        position_dim=16,
        acoustic_dim=12,
        acoustic_embed_dim=32,
        weak_clinical_dim=4,
        clinical_embed_dim=16,
        segment_embedding_dim=128,
        patient_embedding_dim=160,
        num_positions=4,
        position_embedding_dropout=0.0,
        use_murmur_evidence=False,
        murmur_evidence_threshold=0.50,
    ):
        super().__init__()
        self.use_murmur_evidence = use_murmur_evidence
        self.murmur_evidence_threshold = murmur_evidence_threshold
        self.scale1_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale2_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale3_encoder = ScaleEncoder(out_dim=branch_dim)
        self.position_embedding = nn.Embedding(num_positions, position_dim)
        self.position_embedding_dropout = nn.Dropout(position_embedding_dropout)

        self.acoustic_head = nn.Sequential(
            nn.Linear(acoustic_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, acoustic_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.weak_clinical_head = nn.Sequential(
            nn.Linear(weak_clinical_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(24, clinical_embed_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = branch_dim * 3 + position_dim + acoustic_embed_dim
        self.segment_head = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.LayerNorm(192),
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
        self.segment_outcome_head = nn.Linear(segment_embedding_dim, 2)
        self.segment_murmur_head = nn.Linear(segment_embedding_dim, 2)
        self.murmur_evidence_dim = 18 if use_murmur_evidence else 0
        self.patient_head = nn.Sequential(
            nn.Linear(segment_embedding_dim * 2 + clinical_embed_dim + self.murmur_evidence_dim, patient_embedding_dim),
            nn.LayerNorm(patient_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(patient_embedding_dim, patient_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.outcome_head = nn.Linear(patient_embedding_dim, 2)

    def _encode_segments(self, x_scale1, x_scale2, x_scale3, acoustic, position_index):
        batch_size, num_segments = x_scale1.shape[:2]
        flat_x1 = x_scale1.reshape(batch_size * num_segments, *x_scale1.shape[2:])
        flat_x2 = x_scale2.reshape(batch_size * num_segments, *x_scale2.shape[2:])
        flat_x3 = x_scale3.reshape(batch_size * num_segments, *x_scale3.shape[2:])
        flat_acoustic = acoustic.reshape(batch_size * num_segments, acoustic.size(-1))
        flat_positions = position_index.reshape(batch_size * num_segments)

        f1 = self.scale1_encoder(flat_x1)
        f2 = self.scale2_encoder(flat_x2)
        f3 = self.scale3_encoder(flat_x3)
        pos = self.position_embedding_dropout(self.position_embedding(flat_positions.long()))
        acoustic_features = self.acoustic_head(flat_acoustic)
        segment_features = torch.cat([f1, f2, f3, pos, acoustic_features], dim=1)
        segment_embeddings = self.segment_head(segment_features)
        return segment_embeddings.view(batch_size, num_segments, -1)

    def forward(
        self,
        x_scale1,
        x_scale2,
        x_scale3,
        acoustic,
        position_index,
        segment_mask,
        weak_clinical,
        return_embedding=False,
    ):
        segment_embeddings = self._encode_segments(x_scale1, x_scale2, x_scale3, acoustic, position_index)
        mask = segment_mask.float().clamp(0.0, 1.0)
        segment_murmur_logits = self.segment_murmur_head(segment_embeddings)
        attention_logits = self.attention(segment_embeddings).squeeze(-1)
        attention_logits = attention_logits.masked_fill(mask <= 0.0, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1) * mask
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        attentive_pool = torch.sum(segment_embeddings * attention_weights.unsqueeze(-1), dim=1)
        max_pool = segment_embeddings.masked_fill(mask.unsqueeze(-1) <= 0.0, -1e4).max(dim=1).values
        max_pool = torch.where(max_pool < -1e3, torch.zeros_like(max_pool), max_pool)
        clinical_embedding = self.weak_clinical_head(weak_clinical.float())
        patient_inputs = [attentive_pool, max_pool, clinical_embedding]
        murmur_evidence_features = None
        if self.use_murmur_evidence:
            murmur_probs = torch.softmax(segment_murmur_logits, dim=-1)[..., 1]
            murmur_evidence_features = self._murmur_evidence_features(
                murmur_probs,
                position_index,
                mask,
            )
            patient_inputs.append(murmur_evidence_features)
        patient_embedding = self.patient_head(torch.cat(patient_inputs, dim=1))
        logits = self.outcome_head(patient_embedding)
        segment_logits = self.segment_outcome_head(segment_embeddings)

        if return_embedding:
            return {
                "outcome_logits": logits,
                "segment_outcome_logits": segment_logits,
                "segment_murmur_logits": segment_murmur_logits,
                "patient_embedding": patient_embedding,
                "attention_weights": attention_weights,
                "murmur_evidence_features": murmur_evidence_features,
            }
        return logits

    def _masked_topk_mean(self, values, mask, k):
        masked = values.masked_fill(mask <= 0.0, -1e4)
        topk_values = torch.topk(masked, k=min(k, values.size(1)), dim=1).values
        valid_topk = topk_values > -1e3
        topk_sum = torch.where(valid_topk, topk_values, torch.zeros_like(topk_values)).sum(dim=1)
        denom = torch.minimum(mask.sum(dim=1), values.new_full((values.size(0),), float(k))).clamp_min(1.0)
        return topk_sum / denom

    def _murmur_evidence_features(self, murmur_probs, position_index, mask):
        mask = mask.float()
        count = mask.sum(dim=1).clamp_min(1.0)
        masked_probs = murmur_probs * mask
        mean_prob = masked_probs.sum(dim=1) / count
        max_prob = murmur_probs.masked_fill(mask <= 0.0, -1e4).max(dim=1).values
        max_prob = torch.where(max_prob < -1e3, torch.zeros_like(max_prob), max_prob)
        top3 = self._masked_topk_mean(murmur_probs, mask, 3)
        top5 = self._masked_topk_mean(murmur_probs, mask, 5)
        variance = (((murmur_probs - mean_prob.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
        std_prob = torch.sqrt(variance.clamp_min(1e-8))
        soft_evidence = torch.sigmoid((murmur_probs - self.murmur_evidence_threshold) * 10.0) * mask
        high_ratio = soft_evidence.sum(dim=1) / count

        features = [mean_prob, max_prob, top3, top5, std_prob, high_ratio]
        for pos_idx in range(len(POSITIONS)):
            pos_mask = ((position_index == pos_idx).float() * mask)
            pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
            pos_probs = murmur_probs * pos_mask
            pos_mean = pos_probs.sum(dim=1) / pos_count
            pos_max = murmur_probs.masked_fill(pos_mask <= 0.0, -1e4).max(dim=1).values
            pos_max = torch.where(pos_max < -1e3, torch.zeros_like(pos_max), pos_max)
            pos_high = (soft_evidence * pos_mask).sum(dim=1) / pos_count
            features.extend([pos_mean, pos_max, pos_high])
        return torch.stack(features, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-primary outcome model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--max-fn", type=int, default=-1)
    parser.add_argument("--outcome-abnormal-weight", type=float, default=1.0)
    parser.add_argument("--max-segments-per-patient", type=int, default=32)
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--segment-hop", type=float, default=2.0)
    parser.add_argument("--model-type", type=str, default="audio_primary_v1")
    parser.add_argument("--position-embedding-dropout", type=float, default=0.0)
    parser.add_argument("--train-position-dropout-prob", type=float, default=0.0)
    parser.add_argument("--min-positions-after-dropout", type=int, default=1)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sample-normal-absent-weight", type=float, default=1.0)
    parser.add_argument("--sample-complete-normal-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.35)
    parser.add_argument(
        "--hard-negative-scope",
        type=str,
        choices=["all_normal", "absent_normal"],
        default="all_normal",
    )
    parser.add_argument("--hard-negative-absent-normal-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-complete-normal-weight", type=float, default=1.0)
    parser.add_argument("--soft-fn-weight", type=float, default=0.0)
    parser.add_argument("--soft-fn-margin", type=float, default=0.35)
    parser.add_argument("--alpha-segment-outcome", type=float, default=0.0)
    parser.add_argument("--alpha-segment-murmur", type=float, default=0.0)
    parser.add_argument("--segment-murmur-present-weight", type=float, default=2.0)
    parser.add_argument("--use-murmur-evidence", action="store_true")
    parser.add_argument("--murmur-evidence-threshold", type=float, default=0.50)
    parser.add_argument("--segment-evidence-threshold", type=float, default=0.50)
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["auc", "recall_specificity", "fn_fp"],
        default="auc",
    )
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "audio_primary_v1.pth"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_ROOT / "audio_primary_v1"))
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


def score_to_json(score):
    values = []
    for item in score:
        if isinstance(item, (bool, np.bool_)):
            values.append(bool(item))
        elif isinstance(item, (int, np.integer)):
            values.append(int(item))
        elif isinstance(item, (float, np.floating)):
            values.append(float(item))
        else:
            values.append(item)
    return json.dumps(values, ensure_ascii=False)


def outcome_loss(logits, batch, class_weights, args, segment_logits=None, segment_murmur_logits=None):
    ce_loss = F.cross_entropy(logits, batch["y_outcome"], weight=class_weights)
    probs = torch.softmax(logits, dim=1)[:, 1]

    hard_negative_loss = logits.new_tensor(0.0)
    if args.hard_negative_weight > 0.0:
        normal_mask = batch["y_outcome"] == 0
        absent_normal = normal_mask & (batch["y_murmur"] == 0)
        target_normal_mask = absent_normal if args.hard_negative_scope == "absent_normal" else normal_mask
        if target_normal_mask.any():
            normal_penalty = F.relu(probs - args.hard_negative_margin).pow(2)
            normal_weights = torch.ones_like(probs)
            complete_normal = normal_mask & batch["has_all_positions"].bool()
            normal_weights = torch.where(
                absent_normal,
                normal_weights * args.hard_negative_absent_normal_weight,
                normal_weights,
            )
            normal_weights = torch.where(
                complete_normal,
                normal_weights * args.hard_negative_complete_normal_weight,
                normal_weights,
            )
            weighted = normal_penalty[target_normal_mask] * normal_weights[target_normal_mask]
            hard_negative_loss = weighted.mean()

    soft_fn_loss = logits.new_tensor(0.0)
    if args.soft_fn_weight > 0.0:
        abnormal_mask = batch["y_outcome"] == 1
        if abnormal_mask.any():
            soft_fn_loss = F.relu(args.soft_fn_margin - probs[abnormal_mask]).pow(2).mean()

    segment_loss = logits.new_tensor(0.0)
    if args.alpha_segment_outcome > 0.0 and segment_logits is not None:
        batch_size, num_segments = segment_logits.shape[:2]
        flat_logits = segment_logits.reshape(batch_size * num_segments, -1)
        flat_labels = batch["y_outcome"].unsqueeze(1).expand(batch_size, num_segments).reshape(-1)
        flat_mask = batch["segment_mask"].float().reshape(-1)
        flat_loss = F.cross_entropy(flat_logits, flat_labels, weight=class_weights, reduction="none")
        segment_loss = (flat_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

    segment_murmur_loss = logits.new_tensor(0.0)
    if args.alpha_segment_murmur > 0.0 and segment_murmur_logits is not None:
        known_murmur = (batch["y_murmur"] == 0) | (batch["y_murmur"] == 1)
        if known_murmur.any():
            batch_size, num_segments = segment_murmur_logits.shape[:2]
            flat_logits = segment_murmur_logits.reshape(batch_size * num_segments, -1)
            flat_labels = batch["y_murmur"].clamp(0, 1).unsqueeze(1).expand(batch_size, num_segments).reshape(-1)
            flat_segment_mask = batch["segment_mask"].float().reshape(-1)
            flat_known = known_murmur.unsqueeze(1).expand(batch_size, num_segments).reshape(-1).float()
            flat_mask = flat_segment_mask * flat_known
            murmur_weights = torch.tensor(
                [1.0, args.segment_murmur_present_weight],
                dtype=torch.float32,
                device=logits.device,
            )
            flat_loss = F.cross_entropy(flat_logits, flat_labels, weight=murmur_weights, reduction="none")
            segment_murmur_loss = (flat_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

    return (
        ce_loss
        + args.hard_negative_weight * hard_negative_loss
        + args.soft_fn_weight * soft_fn_loss
        + args.alpha_segment_outcome * segment_loss
        + args.alpha_segment_murmur * segment_murmur_loss
    )


def train_one_epoch(model, loader, optimizer, device, class_weights, args):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["x_scale1"],
            batch["x_scale2"],
            batch["x_scale3"],
            batch["acoustic"],
            batch["position_index"],
            batch["segment_mask"],
            batch["weak_clinical"],
            return_embedding=True,
        )
        loss = outcome_loss(
            outputs["outcome_logits"],
            batch,
            class_weights,
            args,
            segment_logits=outputs.get("segment_outcome_logits"),
            segment_murmur_logits=outputs.get("segment_murmur_logits"),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def auc_metrics_from_df(df):
    y_true = df["y_outcome"].to_numpy(dtype=np.int64)
    probs = df["prob_abnormal"].to_numpy(dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return None, None
    return float(roc_auc_score(y_true, probs)), float(average_precision_score(y_true, probs))


def subgroup_key(patient):
    position_bucket = "complete" if patient["has_all_positions"] else f"pos{patient['position_count']}"
    return (patient["y_outcome"], patient["y_murmur"], position_bucket)


def make_balanced_sampler(dataset, indices, args):
    keys = [subgroup_key(dataset.patients[index]) for index in indices]
    counts = Counter(keys)
    weights = []
    for index, key in zip(indices, keys):
        patient = dataset.patients[index]
        weight = 1.0 / counts[key]
        if patient["y_outcome"] == 0 and patient["y_murmur"] == 0:
            weight *= args.sample_normal_absent_weight
        if patient["y_outcome"] == 0 and patient["has_all_positions"]:
            weight *= args.sample_complete_normal_weight
        weights.append(weight)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def collect_predictions(
    model,
    loader,
    dataset,
    device,
    segment_evidence_threshold=0.50,
    evidence_score_source="segment_outcome_head",
):
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
                moved["acoustic"],
                moved["position_index"],
                moved["segment_mask"],
                moved["weak_clinical"],
                return_embedding=True,
            )
            probs = torch.softmax(outputs["outcome_logits"], dim=1)[:, 1].detach().cpu().numpy()
            segment_outcome_probs = torch.softmax(outputs["segment_outcome_logits"], dim=-1)[..., 1].detach().cpu().numpy()
            segment_murmur_probs = torch.softmax(outputs["segment_murmur_logits"], dim=-1)[..., 1].detach().cpu().numpy()
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
                valid_segment_outcome_probs = segment_outcome_probs[i][valid] if valid.any() else np.asarray([])
                valid_segment_murmur_probs = segment_murmur_probs[i][valid] if valid.any() else np.asarray([])
                evidence_probs = (
                    valid_segment_murmur_probs
                    if evidence_score_source == "segment_murmur_head"
                    else valid_segment_outcome_probs
                )
                attention_by_position = {f"attention_{pos}": 0.0 for pos in POSITIONS}
                evidence_by_position = {f"evidence_{pos}": 0 for pos in POSITIONS}
                murmur_by_position = {f"segment_murmur_mean_{pos}": np.nan for pos in POSITIONS}
                murmur_max_by_position = {f"segment_murmur_max_{pos}": np.nan for pos in POSITIONS}
                top_attention_position = ""
                top_attention_weight = np.nan
                segment_mean_prob = np.nan
                segment_max_prob = np.nan
                segment_murmur_mean_prob = np.nan
                segment_murmur_max_prob = np.nan
                evidence_segment_count = 0
                evidence_position_count = 0
                if len(weights) > 0:
                    top_idx = int(np.argmax(weights))
                    top_attention_position = POSITIONS[int(pos_values[top_idx])]
                    top_attention_weight = float(weights[top_idx])
                    segment_mean_prob = float(valid_segment_outcome_probs.mean())
                    segment_max_prob = float(valid_segment_outcome_probs.max())
                    segment_murmur_mean_prob = float(valid_segment_murmur_probs.mean())
                    segment_murmur_max_prob = float(valid_segment_murmur_probs.max())
                    evidence_mask = evidence_probs >= segment_evidence_threshold
                    evidence_segment_count = int(evidence_mask.sum())
                    for pos_idx, pos_name in enumerate(POSITIONS):
                        attention_by_position[f"attention_{pos_name}"] = float(weights[pos_values == pos_idx].sum())
                        evidence_count = int(evidence_mask[pos_values == pos_idx].sum())
                        evidence_by_position[f"evidence_{pos_name}"] = evidence_count
                        pos_murmur = valid_segment_murmur_probs[pos_values == pos_idx]
                        if pos_murmur.size:
                            murmur_by_position[f"segment_murmur_mean_{pos_name}"] = float(pos_murmur.mean())
                            murmur_max_by_position[f"segment_murmur_max_{pos_name}"] = float(pos_murmur.max())
                    evidence_position_count = int(
                        sum(1 for pos_name in POSITIONS if evidence_by_position[f"evidence_{pos_name}"] > 0)
                    )

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
                    "segment_mean_prob": segment_mean_prob,
                    "segment_max_prob": segment_max_prob,
                    "segment_murmur_mean_prob": segment_murmur_mean_prob,
                    "segment_murmur_max_prob": segment_murmur_max_prob,
                    "evidence_score_source": evidence_score_source,
                    "evidence_segment_count": evidence_segment_count,
                    "evidence_position_count": evidence_position_count,
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
                row.update(evidence_by_position)
                row.update(murmur_by_position)
                row.update(murmur_max_by_position)
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate(model, loader, dataset, device, target_recall, args):
    evidence_score_source = (
        "segment_murmur_head" if args.use_murmur_evidence or args.alpha_segment_murmur > 0.0 else "segment_outcome_head"
    )
    df = collect_predictions(
        model,
        loader,
        dataset,
        device,
        args.segment_evidence_threshold,
        evidence_score_source=evidence_score_source,
    )
    selected = search_recall_threshold(
        df["y_outcome"].to_numpy(dtype=np.int64),
        df["prob_abnormal"].to_numpy(dtype=np.float32),
        target_recall,
    )
    roc_auc, pr_auc = auc_metrics_from_df(df)
    selected["roc_auc"] = roc_auc
    selected["pr_auc"] = pr_auc
    return selected


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print("Using device:", device)
    train_dataset = CirCorAudioPrimaryDataset(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        segment_duration=args.segment_duration,
        segment_hop=args.segment_hop,
        max_segments_per_patient=args.max_segments_per_patient,
        training=True,
        position_dropout_prob=args.train_position_dropout_prob,
        min_positions_after_dropout=args.min_positions_after_dropout,
    )
    val_dataset = CirCorAudioPrimaryDataset(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        segment_duration=args.segment_duration,
        segment_hop=args.segment_hop,
        max_segments_per_patient=args.max_segments_per_patient,
        training=False,
    )
    indices = list(range(len(val_dataset)))
    labels = [val_dataset.patients[index]["y_outcome"] for index in indices]
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    print("Train outcome counts:", dict(Counter(val_dataset.patients[idx]["y_outcome"] for idx in train_indices)))
    print("Val outcome counts:", dict(Counter(val_dataset.patients[idx]["y_outcome"] for idx in val_indices)))
    print("Train patients:", len(train_indices))
    print("Val patients:", len(val_indices))

    train_sampler = make_balanced_sampler(train_dataset, train_indices, args) if args.balanced_sampler else None
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=audio_primary_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=audio_primary_collate,
        pin_memory=device.type == "cuda",
    )
    model = AudioPrimaryOutcomeModel(
        acoustic_dim=val_dataset.acoustic_dim,
        weak_clinical_dim=val_dataset.weak_clinical_dim,
        position_embedding_dropout=args.position_embedding_dropout,
        use_murmur_evidence=args.use_murmur_evidence,
        murmur_evidence_threshold=args.murmur_evidence_threshold,
    ).to(device)
    class_weights = torch.tensor([1.0, args.outcome_abnormal_weight], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = None
    best_state = None
    best_metric = None
    best_epoch = None
    epoch_history = []
    patience = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights, args)
        metric = evaluate(model, val_loader, val_dataset, device, args.target_recall, args)
        roc_auc_score_value = metric["roc_auc"] if metric["roc_auc"] is not None else 0.0
        pr_auc_score_value = metric["pr_auc"] if metric["pr_auc"] is not None else 0.0
        if args.selection_mode == "fn_fp":
            fn_target_met = args.max_fn < 0 or metric["fn"] <= args.max_fn
            if fn_target_met:
                score = (
                    1,
                    -metric["fp"],
                    -metric["fn"],
                    metric["specificity"],
                    metric["abnormal_recall"],
                    metric["accuracy"],
                    roc_auc_score_value,
                    pr_auc_score_value,
                )
            else:
                score = (
                    0,
                    -metric["fn"],
                    -metric["fp"],
                    metric["abnormal_recall"],
                    metric["specificity"],
                    metric["accuracy"],
                    roc_auc_score_value,
                    pr_auc_score_value,
                )
        elif args.selection_mode == "recall_specificity":
            score = (
                metric["abnormal_recall"] >= args.target_recall,
                metric["specificity"],
                roc_auc_score_value,
                pr_auc_score_value,
                metric["accuracy"],
            )
        else:
            score = (
                roc_auc_score_value,
                pr_auc_score_value,
                metric["specificity"],
                metric["accuracy"],
            )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f} | "
            f"auc={metric['roc_auc']:.4f} | pr_auc={metric['pr_auc']:.4f} | "
            f"threshold={metric['threshold']:.2f} | acc={metric['accuracy']:.4f} | "
            f"recall={metric['abnormal_recall']:.4f} | specificity={metric['specificity']:.4f} | "
            f"FN={metric['fn']} | FP={metric['fp']}"
        )
        improved = best_score is None or score > best_score
        history_row = {
            "epoch": epoch + 1,
            "loss": float(loss),
            "selection_mode": args.selection_mode,
            "score": score_to_json(score),
            "is_best": bool(improved),
        }
        history_row.update({key: value for key, value in metric.items() if key not in {"roc_auc", "pr_auc"}})
        history_row["roc_auc"] = metric["roc_auc"]
        history_row["pr_auc"] = metric["pr_auc"]
        history_row["target_recall"] = args.target_recall
        history_row["max_fn"] = args.max_fn
        epoch_history.append(history_row)

        if improved:
            best_score = score
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience = 0
            print(f"New best audio-primary checkpoint by {args.selection_mode}.")
        else:
            patience += 1
            print(f"No {args.selection_mode} improvement for {patience} epoch(s).")
        if patience >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in epoch_history:
        row["is_final_best"] = row["epoch"] == best_epoch
    epoch_history_df = pd.DataFrame(epoch_history)
    epoch_history_path = output_dir / "epoch_history.csv"
    epoch_history_json_path = output_dir / "epoch_history.json"
    epoch_history_df.to_csv(epoch_history_path, index=False, encoding="utf-8-sig")
    epoch_history_json_path.write_text(
        json.dumps(epoch_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "args": vars(args),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "model_type": args.model_type,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    evidence_score_source = (
        "segment_murmur_head" if args.use_murmur_evidence or args.alpha_segment_murmur > 0.0 else "segment_outcome_head"
    )
    pred_df = collect_predictions(
        model,
        val_loader,
        val_dataset,
        device,
        args.segment_evidence_threshold,
        evidence_score_source=evidence_score_source,
    )
    selection_notes = {
        "auc": "Checkpoint selected by ROC-AUC, then PR-AUC, specificity, and accuracy.",
        "recall_specificity": "Checkpoint selected by target-recall feasibility, then specificity, ROC-AUC, PR-AUC, and accuracy.",
        "fn_fp": "Checkpoint selected by FN/FP business objective: before FN target is met, reduce FN first; after FN target is met, reduce FP first.",
    }
    summary = write_diagnostics(
        pred_df,
        args.output_dir,
        target_recall=args.target_recall,
        extra_summary={
            "model_type": args.model_type,
            "checkpoint_path": str(checkpoint_path),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_history_path": str(epoch_history_path),
            "epoch_history_json_path": str(epoch_history_json_path),
            "selection_note": selection_notes.get(args.selection_mode, args.selection_mode),
            "feature_note": (
                "Audio-first model using log-mel segments, auscultation position embedding, handcrafted acoustic "
                "descriptors, and weak clinical values age/sex/height/weight. It excludes Murmur, pregnancy status, "
                "position_count, has_all_positions, per-position presence flags, and clinical missingness flags."
            ),
            "training_strategy_note": (
                "Optional controls: training-time position dropout, outcome/murmur/position subgroup sampler, "
                "position embedding dropout, scoped hard-negative penalty for high-scoring normal samples, and "
                "soft false-negative penalty for low-scoring abnormal samples. Segment Murmur auxiliary supervision "
                "and Murmur-evidence aggregation can be enabled for audio-derived evidence experiments. "
                "See saved args for enabled settings."
            ),
            "segment_config": {
                "segment_duration_sec": args.segment_duration,
                "segment_hop_sec": args.segment_hop,
                "max_segments_per_patient": args.max_segments_per_patient,
            },
        },
    )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved checkpoint:", checkpoint_path)
    print("Saved metadata:", metadata_path)
    print("Wrote diagnostics:", args.output_dir)


if __name__ == "__main__":
    main()
