import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset_hms import CirCorHMSDataset
from hms_feature_utils import blend_patient_probabilities
from model_hms import HMSLiteModel
from paths import CHECKPOINT_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMS murmur classifier.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--alpha-t", type=float, default=0.10)
    parser.add_argument("--alpha-g", type=float, default=0.10)
    parser.add_argument("--alpha-s", type=float, default=0.05)
    parser.add_argument("--patient-loss-weight", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0)
    parser.add_argument("--spec-time-mask", type=int, default=0)
    parser.add_argument("--spec-freq-mask", type=int, default=0)
    parser.add_argument("--spec-num-masks", type=int, default=1)
    parser.add_argument("--decision-weight-absent", type=float, default=1.0)
    parser.add_argument("--decision-weight-present", type=float, default=1.0)
    parser.add_argument("--decision-weight-unknown", type=float, default=1.0)
    parser.add_argument("--disable-patient-scale-search", action="store_true")
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "best_model_hms.pth"))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_specaugment_batch(x, time_mask_width=10, freq_mask_width=6, num_masks=1):
    if num_masks <= 0 or (time_mask_width <= 0 and freq_mask_width <= 0):
        return x

    augmented = x.clone()
    _, _, freq_bins, time_steps = augmented.shape

    for batch_index in range(augmented.size(0)):
        for _ in range(num_masks):
            if freq_mask_width > 0 and freq_bins > 1:
                width = random.randint(0, min(freq_mask_width, freq_bins - 1))
                if width > 0:
                    start = random.randint(0, freq_bins - width)
                    augmented[batch_index, :, start : start + width, :] = 0.0

            if time_mask_width > 0 and time_steps > 1:
                width = random.randint(0, min(time_mask_width, time_steps - 1))
                if width > 0:
                    start = random.randint(0, time_steps - width)
                    augmented[batch_index, :, :, start : start + width] = 0.0

    return augmented


def compute_patient_bag_loss(murmur_logits, y_murmur, patient_ids, class_weights=None):
    if len(patient_ids) <= 1:
        return None

    grouped_indices = defaultdict(list)
    for sample_index, patient_id in enumerate(patient_ids):
        grouped_indices[str(patient_id)].append(sample_index)

    if not grouped_indices:
        return None

    probs = torch.softmax(murmur_logits, dim=1)
    patient_log_probs = []
    patient_targets = []

    for indices in grouped_indices.values():
        index_tensor = torch.tensor(indices, device=murmur_logits.device, dtype=torch.long)
        mean_prob = probs.index_select(0, index_tensor).mean(dim=0).clamp_min(1e-8)
        patient_log_probs.append(mean_prob.log())
        patient_targets.append(y_murmur[index_tensor[0]])

    patient_log_probs = torch.stack(patient_log_probs, dim=0)
    patient_targets = torch.stack(patient_targets, dim=0)
    return F.nll_loss(patient_log_probs, patient_targets, weight=class_weights)


def create_ema_model(model):
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model, model, decay):
    if ema_model is None:
        return

    with torch.no_grad():
        model_state = model.state_dict()
        for key, ema_value in ema_model.state_dict().items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.copy_(decay * ema_value + (1.0 - decay) * model_value)
            else:
                ema_value.copy_(model_value)


def compute_weighted_accuracy(y_true, y_pred):
    weight_map = {
        1: 5.0,  # Present
        2: 3.0,  # Unknown
        0: 1.0,  # Absent
    }

    weighted_correct = 0.0
    weighted_total = 0.0
    for yt, yp in zip(y_true, y_pred):
        weight = weight_map[int(yt)]
        weighted_total += weight
        if int(yt) == int(yp):
            weighted_correct += weight

    return weighted_correct / weighted_total if weighted_total > 0 else 0.0


def predict_metric_aligned_classes(prob_tensor, decision_weights):
    if prob_tensor.ndim == 1:
        prob_tensor = prob_tensor.unsqueeze(0)
    local_weights = decision_weights.to(prob_tensor.device)
    weighted_scores = prob_tensor * local_weights.view(1, -1)
    return torch.argmax(weighted_scores, dim=1)


def search_best_patient_decision_weights(patient_probs, patient_labels, base_decision_weights):
    fused_probs = torch.stack(patient_probs, dim=0)
    labels = np.asarray(patient_labels, dtype=np.int64)
    base_weights = base_decision_weights.detach().cpu().float()

    present_scales = [0.9, 1.0, 1.1, 1.2]
    unknown_scales = [0.8, 1.0, 1.25, 1.5, 1.8, 2.1, 2.5]

    best_payload = None
    for present_scale in present_scales:
        for unknown_scale in unknown_scales:
            candidate_weights = base_weights.clone()
            candidate_weights[1] = candidate_weights[1] * present_scale
            candidate_weights[2] = candidate_weights[2] * unknown_scale

            preds = predict_metric_aligned_classes(fused_probs, candidate_weights).cpu().numpy().astype(np.int64)
            weighted_acc = compute_weighted_accuracy(labels, preds)

            if best_payload is None or weighted_acc > best_payload["weighted_acc"]:
                best_payload = {
                    "weighted_acc": float(weighted_acc),
                    "decision_weights": candidate_weights,
                    "preds": preds.tolist(),
                }

    return best_payload


def masked_ce_loss(logits, targets, mask, label_smoothing=0.0):
    if logits is None:
        return None

    valid = mask > 0.5
    if valid.sum().item() == 0:
        return None

    return F.cross_entropy(
        logits[valid],
        targets[valid],
        label_smoothing=label_smoothing,
    )


def train_one_epoch(
    model,
    loader,
    murmur_criterion,
    optimizer,
    device,
    class_weights=None,
    alpha_t=0.10,
    alpha_g=0.10,
    alpha_s=0.05,
    patient_loss_weight=0.20,
    ema_model=None,
    ema_decay=0.995,
    spec_time_mask=10,
    spec_freq_mask=6,
    spec_num_masks=1,
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        x1 = batch["x_scale1"].to(device)
        x2 = batch["x_scale2"].to(device)
        x3 = batch["x_scale3"].to(device)
        position_index = batch["position_index"].to(device)
        patient_ids = batch["patient_id"]

        y_murmur = batch["y_murmur"].to(device)
        y_timing = batch["y_timing"].to(device)
        timing_mask = batch["timing_mask"].to(device)
        y_grade = batch["y_grade"].to(device)
        grade_mask = batch["grade_mask"].to(device)
        y_shape = batch["y_shape"].to(device)
        shape_mask = batch["shape_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        x1 = apply_specaugment_batch(
            x1,
            time_mask_width=spec_time_mask,
            freq_mask_width=spec_freq_mask,
            num_masks=spec_num_masks,
        )
        x2 = apply_specaugment_batch(
            x2,
            time_mask_width=spec_time_mask,
            freq_mask_width=spec_freq_mask,
            num_masks=spec_num_masks,
        )
        x3 = apply_specaugment_batch(
            x3,
            time_mask_width=spec_time_mask,
            freq_mask_width=spec_freq_mask,
            num_masks=spec_num_masks,
        )

        murmur_logits, timing_logits, grade_logits, shape_logits = model(
            x1,
            x2,
            x3,
            position_index=position_index,
        )

        loss = murmur_criterion(murmur_logits, y_murmur)

        loss_timing = masked_ce_loss(timing_logits, y_timing, timing_mask, label_smoothing=0.0)
        if loss_timing is not None:
            loss = loss + alpha_t * loss_timing

        loss_grade = masked_ce_loss(grade_logits, y_grade, grade_mask, label_smoothing=0.0)
        if loss_grade is not None:
            loss = loss + alpha_g * loss_grade

        loss_shape = masked_ce_loss(shape_logits, y_shape, shape_mask, label_smoothing=0.0)
        if loss_shape is not None:
            loss = loss + alpha_s * loss_shape

        patient_loss = compute_patient_bag_loss(
            murmur_logits,
            y_murmur,
            patient_ids=patient_ids,
            class_weights=class_weights,
        )
        if patient_loss is not None and patient_loss_weight > 0.0:
            loss = loss + patient_loss_weight * patient_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        update_ema_model(ema_model, model, decay=ema_decay)

        total_loss += loss.item()

        preds = torch.argmax(murmur_logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y_murmur.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    seg_acc = accuracy_score(all_labels, all_preds)
    seg_wacc = compute_weighted_accuracy(all_labels, all_preds)
    return avg_loss, seg_acc, seg_wacc


def evaluate_patient_level(model, loader, murmur_criterion, device, decision_weights, search_patient_scales=False):
    model.eval()

    total_loss = 0.0
    seg_preds = []
    seg_labels = []
    patient_probs = defaultdict(list)
    patient_true = {}

    with torch.no_grad():
        for batch in loader:
            x1 = batch["x_scale1"].to(device)
            x2 = batch["x_scale2"].to(device)
            x3 = batch["x_scale3"].to(device)
            position_index = batch["position_index"].to(device)
            y_murmur = batch["y_murmur"].to(device)
            patient_ids = batch["patient_id"]

            murmur_logits, _, _, _ = model(
                x1,
                x2,
                x3,
                position_index=position_index,
            )

            loss = murmur_criterion(murmur_logits, y_murmur)
            total_loss += loss.item()

            probs = torch.softmax(murmur_logits, dim=1)
            preds = predict_metric_aligned_classes(probs, decision_weights)

            seg_preds.extend(preds.cpu().numpy().tolist())
            seg_labels.extend(y_murmur.cpu().numpy().tolist())

            for idx, patient_id in enumerate(patient_ids):
                patient_probs[patient_id].append(probs[idx].cpu())
                patient_true[patient_id] = int(y_murmur[idx].cpu().item())

    avg_loss = total_loss / max(len(loader), 1)
    seg_acc = accuracy_score(seg_labels, seg_preds)
    seg_wacc = compute_weighted_accuracy(seg_labels, seg_preds)

    patient_labels = []
    patient_preds = []
    patient_fused_probs = []
    for patient_id in sorted(patient_probs.keys()):
        fused_prob = blend_patient_probabilities(patient_probs[patient_id])
        patient_fused_probs.append(fused_prob)
        patient_labels.append(patient_true[patient_id])

    tuned_decision_weights = decision_weights.detach().cpu()
    if search_patient_scales and patient_fused_probs:
        best_payload = search_best_patient_decision_weights(
            patient_fused_probs,
            patient_labels,
            decision_weights,
        )
        tuned_decision_weights = best_payload["decision_weights"]
        patient_preds = best_payload["preds"]
    else:
        for fused_prob in patient_fused_probs:
            patient_preds.append(int(predict_metric_aligned_classes(fused_prob, decision_weights)[0].item()))

    patient_acc = accuracy_score(patient_labels, patient_preds)
    patient_wacc = compute_weighted_accuracy(patient_labels, patient_preds)
    return (
        avg_loss,
        seg_acc,
        seg_wacc,
        patient_acc,
        patient_wacc,
        patient_labels,
        patient_preds,
        tuned_decision_weights,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Seed:", args.seed)
    print(
        "HMS tuning:",
        {
            "alpha_t": args.alpha_t,
            "alpha_g": args.alpha_g,
            "alpha_s": args.alpha_s,
            "patient_loss_weight": args.patient_loss_weight,
            "ema_decay": args.ema_decay,
            "label_smoothing": args.label_smoothing,
            "spec_time_mask": args.spec_time_mask,
            "spec_freq_mask": args.spec_freq_mask,
            "spec_num_masks": args.spec_num_masks,
            "decision_weight_absent": args.decision_weight_absent,
            "decision_weight_present": args.decision_weight_present,
            "decision_weight_unknown": args.decision_weight_unknown,
            "search_patient_scales": not args.disable_patient_scale_search,
        },
    )

    csv_path = Path(args.csv_path).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}\n"
            "Place training_data.csv under model_train/data/circor/ or override --csv-path."
        )
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"Training wav directory not found: {wav_dir}\n"
            "Place wav files under model_train/data/circor/training_data/ or override --wav-dir."
        )

    print("Training CSV:", csv_path)
    print("Training WAV dir:", wav_dir)

    dataset = CirCorHMSDataset(
        csv_path=csv_path,
        wav_dir=wav_dir,
        sr=2000,
        segment_duration=3.0,
        segment_hop=2.0,
        n_mels=64,
    )

    patient_to_label = {}
    for sample in dataset.samples:
        patient_to_label[sample["patient_id"]] = sample["y_murmur"]

    patient_ids = list(patient_to_label.keys())
    patient_labels = [patient_to_label[patient_id] for patient_id in patient_ids]

    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=0.2,
        random_state=args.seed,
        stratify=patient_labels,
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)

    train_indices = [idx for idx, sample in enumerate(dataset.samples) if sample["patient_id"] in train_patients]
    val_indices = [idx for idx, sample in enumerate(dataset.samples) if sample["patient_id"] in val_patients]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print("Train patients:", len(train_patients))
    print("Val patients:", len(val_patients))
    print("Train segments:", len(train_dataset))
    print("Val segments:", len(val_dataset))

    patient_label_counter = Counter(patient_to_label[patient_id] for patient_id in train_patients)
    class_weights = torch.tensor(
        [
            max(patient_label_counter[0], 1),
            max(patient_label_counter[1], 1),
            max(patient_label_counter[2], 1),
        ],
        dtype=torch.float32,
        device=device,
    )
    class_weights = torch.sqrt(class_weights[0] / class_weights)
    class_weights = torch.clamp(class_weights, min=1.0, max=3.0)
    print("Class weights:", class_weights)

    decision_weights = torch.tensor(
        [
            args.decision_weight_absent,
            args.decision_weight_present,
            args.decision_weight_unknown,
        ],
        dtype=torch.float32,
        device=device,
    )
    print("Decision weights:", decision_weights)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = HMSLiteModel(
        num_timing_classes=dataset.num_timing_classes,
        num_grade_classes=dataset.num_grade_classes,
        num_shape_classes=dataset.num_shape_classes,
        branch_dim=64,
        embedding_dim=128,
        murmur_classes=3,
    ).to(device)
    ema_model = create_ema_model(model) if args.ema_decay > 0.0 else None

    murmur_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    epochs = args.epochs
    patience = args.patience
    patience_counter = 0
    best_patient_wacc = 0.0

    best_model_path = Path(args.checkpoint_path)
    save_dir = best_model_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_seg_acc, train_seg_wacc = train_one_epoch(
            model=model,
            loader=train_loader,
            murmur_criterion=murmur_criterion,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            alpha_t=args.alpha_t,
            alpha_g=args.alpha_g,
            alpha_s=args.alpha_s,
            patient_loss_weight=args.patient_loss_weight,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            spec_time_mask=args.spec_time_mask,
            spec_freq_mask=args.spec_freq_mask,
            spec_num_masks=args.spec_num_masks,
        )

        eval_model = ema_model if ema_model is not None else model

        (
            val_loss,
            val_seg_acc,
            val_seg_wacc,
            val_patient_acc,
            val_patient_wacc,
            val_patient_labels,
            val_patient_preds,
            tuned_decision_weights,
        ) = evaluate_patient_level(
            model=eval_model,
            loader=val_loader,
            murmur_criterion=murmur_criterion,
            device=device,
            decision_weights=decision_weights,
            search_patient_scales=not args.disable_patient_scale_search,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Seg Acc: {train_seg_acc:.4f} | "
            f"Train Seg W.Acc: {train_seg_wacc:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Val Seg Acc: {val_seg_acc:.4f} | "
            f"Val Seg W.Acc: {val_seg_wacc:.4f} | "
            f"Val Patient Acc: {val_patient_acc:.4f} | "
            f"Val Patient W.Acc: {val_patient_wacc:.4f} | "
            f"LR: {lr_now:.6f}"
        )

        if val_patient_wacc > best_patient_wacc:
            best_patient_wacc = val_patient_wacc
            patience_counter = 0
            torch.save(copy.deepcopy(eval_model.state_dict()), best_model_path)
            print(f"Best model saved to: {best_model_path}")
            meta_path = best_model_path.with_suffix(".json")
            meta_payload = {
                "decision_weights": [float(x) for x in tuned_decision_weights.tolist()],
                "best_val_patient_wacc": float(best_patient_wacc),
                "seed": int(args.seed),
            }
            meta_path.write_text(json.dumps(meta_payload, ensure_ascii=True, indent=2), encoding="utf-8")
            print(f"Saved decision metadata to: {meta_path}")
            print("Best decision weights:", [round(float(x), 4) for x in tuned_decision_weights.tolist()])

            label_order = [1, 2, 0]
            cm = confusion_matrix(val_patient_labels, val_patient_preds, labels=label_order)
            print("Patient-level Confusion Matrix (Present, Unknown, Absent):")
            print(cm)

            report = classification_report(
                val_patient_labels,
                val_patient_preds,
                labels=label_order,
                target_names=["Present", "Unknown", "Absent"],
                digits=4,
                zero_division=0,
            )
            print("Patient-level Classification Report:")
            print(report)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished.")
    print(f"Best Val Patient W.Acc: {best_patient_wacc:.4f}")


if __name__ == "__main__":
    main()
