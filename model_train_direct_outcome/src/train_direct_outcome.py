import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset_direct_outcome import CirCorDirectOutcomeDataset, direct_outcome_collate
from model_direct_outcome import DirectOutcomeMultiTaskModel
from paths import CHECKPOINT_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train direct patient-level outcome model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "best_direct_outcome.pth"))
    parser.add_argument("--max-segments-per-patient", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--complete-target-recall-delta",
        type=float,
        default=0.03,
        help="Allowed recall slack for the complete-position validation subset.",
    )
    parser.add_argument(
        "--absent-abnormal-target-recall-delta",
        type=float,
        default=0.03,
        help="Allowed recall slack for Abnormal patients with Murmur=Absent.",
    )
    parser.add_argument("--outcome-abnormal-weight", type=float, default=3.0)
    parser.add_argument(
        "--absent-abnormal-weight",
        type=float,
        default=1.0,
        help="Extra outcome-loss multiplier for Abnormal patients with Murmur=Absent.",
    )
    parser.add_argument("--fn-penalty-weight", type=float, default=0.50)
    parser.add_argument(
        "--fp-penalty-weight",
        type=float,
        default=0.0,
        help="Soft penalty for assigning high Abnormal probability to true Normal patients.",
    )
    parser.add_argument("--alpha-murmur", type=float, default=0.25)
    parser.add_argument("--alpha-timing", type=float, default=0.05)
    parser.add_argument("--alpha-grade", type=float, default=0.05)
    parser.add_argument("--alpha-shape", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic forward/loss check only.")
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


def masked_ce_loss(logits, targets, mask):
    if logits is None:
        return None
    valid = mask > 0.5
    if valid.sum().item() == 0:
        return None
    return F.cross_entropy(logits[valid], targets[valid])


def direct_multitask_loss(
    outputs,
    batch,
    outcome_class_weights,
    label_smoothing=0.02,
    fn_penalty_weight=0.50,
    fp_penalty_weight=0.0,
    absent_abnormal_weight=1.0,
    alpha_murmur=0.25,
    alpha_timing=0.05,
    alpha_grade=0.05,
    alpha_shape=0.03,
):
    y_outcome = batch["y_outcome"]
    outcome_logits = outputs["outcome_logits"]
    outcome_ce = F.cross_entropy(
        outcome_logits,
        y_outcome,
        weight=outcome_class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if absent_abnormal_weight > 1.0:
        absent_abnormal_mask = (y_outcome == 1) & (batch["y_murmur"] == 0)
        sample_weights = torch.ones_like(outcome_ce)
        sample_weights = sample_weights.masked_fill(absent_abnormal_mask, absent_abnormal_weight)
        loss = (outcome_ce * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    else:
        loss = outcome_ce.mean()

    # Screening-first penalty: for true Abnormal patients, explicitly penalize
    # probability mass assigned to Normal. This moves optimization toward lower
    # false negatives instead of plain accuracy.
    probs = torch.softmax(outcome_logits, dim=1)
    abnormal_mask = y_outcome == 1
    if abnormal_mask.any() and fn_penalty_weight > 0.0:
        soft_fn_penalty = probs[abnormal_mask, 0].mean()
        loss = loss + fn_penalty_weight * soft_fn_penalty

    normal_mask = y_outcome == 0
    if normal_mask.any() and fp_penalty_weight > 0.0:
        soft_fp_penalty = probs[normal_mask, 1].mean()
        loss = loss + fp_penalty_weight * soft_fp_penalty

    if alpha_murmur > 0.0:
        loss_murmur = F.cross_entropy(outputs["murmur_logits"], batch["y_murmur"], label_smoothing=label_smoothing)
        loss = loss + alpha_murmur * loss_murmur

    loss_timing = masked_ce_loss(outputs["timing_logits"], batch["y_timing"], batch["timing_mask"])
    if loss_timing is not None:
        loss = loss + alpha_timing * loss_timing

    loss_grade = masked_ce_loss(outputs["grade_logits"], batch["y_grade"], batch["grade_mask"])
    if loss_grade is not None:
        loss = loss + alpha_grade * loss_grade

    loss_shape = masked_ce_loss(outputs["shape_logits"], batch["y_shape"], batch["shape_mask"])
    if loss_shape is not None:
        loss = loss + alpha_shape * loss_shape

    return loss


def metrics_from_probs(y_true, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "abnormal_precision": float(precision),
        "abnormal_recall": float(recall),
        "abnormal_f1": float(f1),
        "specificity": specificity,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def search_recall_threshold(y_true, probs, target_recall):
    thresholds = np.linspace(0.05, 0.95, 91)
    candidates = [metrics_from_probs(y_true, probs, th) for th in thresholds]
    feasible = [item for item in candidates if item["abnormal_recall"] >= target_recall]
    if feasible:
        # Within the requested recall floor, keep as many normals as possible.
        return max(feasible, key=lambda item: (item["specificity"], item["accuracy"], item["threshold"]))
    # If the target cannot be reached, choose the highest recall first.
    return max(candidates, key=lambda item: (item["abnormal_recall"], item["accuracy"]))


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_one_epoch(model, loader, optimizer, device, outcome_class_weights, args):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["x_scale1"],
            batch["x_scale2"],
            batch["x_scale3"],
            batch["position_index"],
            batch["segment_mask"],
            batch["clinical"],
        )
        loss = direct_multitask_loss(
            outputs,
            batch,
            outcome_class_weights=outcome_class_weights,
            label_smoothing=args.label_smoothing,
            fn_penalty_weight=args.fn_penalty_weight,
            fp_penalty_weight=args.fp_penalty_weight,
            absent_abnormal_weight=args.absent_abnormal_weight,
            alpha_murmur=args.alpha_murmur,
            alpha_timing=args.alpha_timing,
            alpha_grade=args.alpha_grade,
            alpha_shape=args.alpha_shape,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def collect_probs(model, loader, device):
    model.eval()
    y_true = []
    y_murmur = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["x_scale1"],
                batch["x_scale2"],
                batch["x_scale3"],
                batch["position_index"],
                batch["segment_mask"],
                batch["clinical"],
            )
            outcome_probs = torch.softmax(outputs["outcome_logits"], dim=1)[:, 1]
            probs.extend(outcome_probs.cpu().numpy().tolist())
            y_true.extend(batch["y_outcome"].cpu().numpy().tolist())
            y_murmur.extend(batch["y_murmur"].cpu().numpy().tolist())
    y_true = np.asarray(y_true, dtype=np.int64)
    y_murmur = np.asarray(y_murmur, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    return y_true, probs, y_murmur


def evaluate(model, loader, device, target_recall):
    y_true, probs, y_murmur = collect_probs(model, loader, device)
    metrics = search_recall_threshold(y_true, probs, target_recall)
    metrics["murmur_subgroups"] = murmur_subgroup_metrics(y_true, probs, y_murmur, metrics["threshold"])
    return metrics


def subgroup_binary_metrics(y_true, probs, mask, threshold):
    count = int(mask.sum())
    if count == 0:
        return None
    return metrics_from_probs(y_true[mask], probs[mask], threshold)


def murmur_subgroup_metrics(y_true, probs, y_murmur, threshold):
    subgroup_defs = {
        "murmur_absent": y_murmur == 0,
        "murmur_present": y_murmur == 1,
        "murmur_unknown": y_murmur == 2,
        "abnormal_murmur_absent": (y_true == 1) & (y_murmur == 0),
        "abnormal_murmur_present": (y_true == 1) & (y_murmur == 1),
        "abnormal_murmur_unknown": (y_true == 1) & (y_murmur == 2),
    }
    results = {}
    preds = (probs >= threshold).astype(np.int64)
    for name, mask in subgroup_defs.items():
        count = int(mask.sum())
        if count == 0:
            results[name] = {"count": 0}
            continue
        if name.startswith("abnormal_"):
            detected = int(preds[mask].sum())
            results[name] = {
                "count": count,
                "detected": detected,
                "recall": float(detected / max(count, 1)),
                "fn": int(count - detected),
            }
        else:
            results[name] = subgroup_binary_metrics(y_true, probs, mask, threshold)
    return results


def get_subgroup_recall(metrics, subgroup_name):
    subgroup = metrics.get("murmur_subgroups", {}).get(subgroup_name)
    if not subgroup or subgroup.get("count", 0) == 0:
        return 1.0
    return subgroup.get("recall", 0.0)


def get_subgroup_fn(metrics, subgroup_name):
    subgroup = metrics.get("murmur_subgroups", {}).get(subgroup_name)
    if not subgroup:
        return 0
    return int(subgroup.get("fn", 0))


def build_selection_score(metrics, complete_metrics, target_recall, complete_target_recall, absent_abnormal_target_recall):
    full_recall_ok = metrics["abnormal_recall"] >= target_recall
    absent_abnormal_recall = get_subgroup_recall(metrics, "abnormal_murmur_absent")
    absent_abnormal_recall_ok = absent_abnormal_recall >= absent_abnormal_target_recall
    if complete_metrics is None:
        complete_recall_ok = True
        complete_specificity = 0.0
        complete_accuracy = 0.0
    else:
        complete_recall_ok = complete_metrics["abnormal_recall"] >= complete_target_recall
        complete_specificity = complete_metrics["specificity"]
        complete_accuracy = complete_metrics["accuracy"]

    # Priority:
    # 1. Full validation recall reaches target.
    # 2. Complete-position validation recall also reaches its floor.
    # 3. Abnormal patients with Murmur=Absent also reach their recall floor.
    # 4. Once recall gates are satisfied, reduce false positives on full validation.
    # 5. Reduce false positives on complete-position validation.
    # 6. Keep accuracy as a final tie-breaker; raw absent-murmur recall only breaks later ties.
    return (
        full_recall_ok,
        complete_recall_ok,
        absent_abnormal_recall_ok,
        metrics["specificity"],
        complete_specificity,
        metrics["accuracy"],
        complete_accuracy,
        absent_abnormal_recall,
        -metrics["fn"],
        -metrics["fp"],
    )


def run_smoke_test(device):
    model = DirectOutcomeMultiTaskModel(
        clinical_dim=10,
        num_timing_classes=4,
        num_grade_classes=3,
        num_shape_classes=4,
    ).to(device)
    batch = {
        "x_scale1": torch.randn(2, 5, 1, 64, 94, device=device),
        "x_scale2": torch.randn(2, 5, 1, 64, 126, device=device),
        "x_scale3": torch.randn(2, 5, 1, 64, 188, device=device),
        "position_index": torch.tensor([[0, 1, 2, 3, 0], [0, 1, 0, 0, 0]], device=device),
        "segment_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.float32, device=device),
        "clinical": torch.randn(2, 10, device=device),
        "y_outcome": torch.tensor([1, 0], device=device),
        "y_murmur": torch.tensor([1, 0], device=device),
        "y_timing": torch.tensor([0, -1], device=device),
        "timing_mask": torch.tensor([1, 0], dtype=torch.float32, device=device),
        "y_grade": torch.tensor([1, -1], device=device),
        "grade_mask": torch.tensor([1, 0], dtype=torch.float32, device=device),
        "y_shape": torch.tensor([2, -1], device=device),
        "shape_mask": torch.tensor([1, 0], dtype=torch.float32, device=device),
    }
    outputs = model(
        batch["x_scale1"],
        batch["x_scale2"],
        batch["x_scale3"],
        batch["position_index"],
        batch["segment_mask"],
        batch["clinical"],
    )
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32, device=device)
    loss = direct_multitask_loss(
        outputs,
        batch,
        weights,
        alpha_murmur=0.0,
        absent_abnormal_weight=1.5,
        fp_penalty_weight=0.05,
    )
    loss.backward()
    print("Smoke test passed.")
    print("outcome_logits:", tuple(outputs["outcome_logits"].shape))
    print("murmur_logits:", tuple(outputs["murmur_logits"].shape))
    print("loss:", round(float(loss.item()), 4))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print("Using device:", device)

    if args.smoke_test:
        run_smoke_test(device)
        return

    csv_path = Path(args.csv_path).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"Training wav directory not found: {wav_dir}\n"
            "Copy wav files into data/circor/training_data/ or override --wav-dir."
        )

    dataset = CirCorDirectOutcomeDataset(
        csv_path=csv_path,
        wav_dir=wav_dir,
        max_segments_per_patient=args.max_segments_per_patient,
    )
    patient_labels = [patient["y_outcome"] for patient in dataset.patients]
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=patient_labels,
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    val_complete_indices = [idx for idx in val_indices if dataset.patients[idx]["has_all_positions"]]
    val_complete_dataset = Subset(dataset, val_complete_indices)

    train_counter = Counter(dataset.patients[idx]["y_outcome"] for idx in train_indices)
    val_counter = Counter(dataset.patients[idx]["y_outcome"] for idx in val_indices)
    val_complete_counter = Counter(dataset.patients[idx]["y_outcome"] for idx in val_complete_indices)
    train_murmur_outcome_counter = Counter(
        (dataset.patients[idx]["y_outcome"], dataset.patients[idx]["y_murmur"]) for idx in train_indices
    )
    val_murmur_outcome_counter = Counter(
        (dataset.patients[idx]["y_outcome"], dataset.patients[idx]["y_murmur"]) for idx in val_indices
    )
    print("Train outcome counts:", dict(train_counter))
    print("Val outcome counts:", dict(val_counter))
    print("Val patients:", len(val_dataset))
    print("Val complete-position outcome counts:", dict(val_complete_counter))
    print("Val complete-position patients:", len(val_complete_dataset))
    print("Train outcome/murmur counts:", dict(train_murmur_outcome_counter))
    print("Val outcome/murmur counts:", dict(val_murmur_outcome_counter))

    outcome_class_weights = torch.tensor(
        [1.0, args.outcome_abnormal_weight],
        dtype=torch.float32,
        device=device,
    )

    model = DirectOutcomeMultiTaskModel(
        clinical_dim=dataset.clinical_dim,
        num_timing_classes=dataset.num_timing_classes,
        num_grade_classes=dataset.num_grade_classes,
        num_shape_classes=dataset.num_shape_classes,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    )
    val_complete_loader = DataLoader(
        val_complete_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    ) if len(val_complete_dataset) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    best_metric = None
    best_selection_score = None
    best_state = None
    patience_counter = 0
    complete_target_recall = max(0.0, args.target_recall - args.complete_target_recall_delta)
    absent_abnormal_target_recall = max(0.0, args.target_recall - args.absent_abnormal_target_recall_delta)
    print("Complete-position target recall floor:", round(complete_target_recall, 4))
    print("Absent-murmur abnormal target recall floor:", round(absent_abnormal_target_recall, 4))

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, outcome_class_weights, args)
        metrics = evaluate(model, val_loader, device, args.target_recall)
        complete_metrics = None
        if val_complete_loader is not None:
            complete_y_true, complete_probs, complete_y_murmur = collect_probs(model, val_complete_loader, device)
            complete_metrics = metrics_from_probs(complete_y_true, complete_probs, metrics["threshold"])
            complete_metrics["murmur_subgroups"] = murmur_subgroup_metrics(
                complete_y_true,
                complete_probs,
                complete_y_murmur,
                metrics["threshold"],
            )
        scheduler.step()

        absent_abnormal_recall = get_subgroup_recall(metrics, "abnormal_murmur_absent")
        absent_abnormal_fn = get_subgroup_fn(metrics, "abnormal_murmur_absent")
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"threshold={metrics['threshold']:.2f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"recall={metrics['abnormal_recall']:.4f} | "
            f"specificity={metrics['specificity']:.4f} | "
            f"FN={metrics['fn']} | FP={metrics['fp']} | "
            f"absent_abn_recall={absent_abnormal_recall:.4f} | "
            f"absent_abn_FN={absent_abnormal_fn}"
        )
        if complete_metrics is not None:
            complete_absent_abnormal_recall = get_subgroup_recall(complete_metrics, "abnormal_murmur_absent")
            complete_absent_abnormal_fn = get_subgroup_fn(complete_metrics, "abnormal_murmur_absent")
            print(
                "  Complete-position val with full-val threshold | "
                f"acc={complete_metrics['accuracy']:.4f} | "
                f"recall={complete_metrics['abnormal_recall']:.4f} | "
                f"specificity={complete_metrics['specificity']:.4f} | "
                f"FN={complete_metrics['fn']} | FP={complete_metrics['fp']} | "
                f"absent_abn_recall={complete_absent_abnormal_recall:.4f} | "
                f"absent_abn_FN={complete_absent_abnormal_fn}"
            )

        score = build_selection_score(
            metrics=metrics,
            complete_metrics=complete_metrics,
            target_recall=args.target_recall,
            complete_target_recall=complete_target_recall,
            absent_abnormal_target_recall=absent_abnormal_target_recall,
        )
        if best_metric is None or score > best_selection_score:
            best_metric = metrics
            if complete_metrics is not None:
                best_metric["complete_position_metric"] = complete_metrics
            best_metric["selection_score"] = list(score)
            best_metric["complete_target_recall"] = complete_target_recall
            best_metric["absent_abnormal_target_recall"] = absent_abnormal_target_recall
            best_selection_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("New best murmur-stratified recall-constrained checkpoint candidate.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "best_metric": best_metric,
                "target_recall": args.target_recall,
                "complete_target_recall": complete_target_recall,
                "absent_abnormal_target_recall": absent_abnormal_target_recall,
                "outcome_abnormal_weight": args.outcome_abnormal_weight,
                "absent_abnormal_weight": args.absent_abnormal_weight,
                "fn_penalty_weight": args.fn_penalty_weight,
                "fp_penalty_weight": args.fp_penalty_weight,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "alpha_murmur": args.alpha_murmur,
                "alpha_timing": args.alpha_timing,
                "alpha_grade": args.alpha_grade,
                "alpha_shape": args.alpha_shape,
                "seed": args.seed,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved checkpoint:", checkpoint_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
