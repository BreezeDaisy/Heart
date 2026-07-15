
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_hms import CirCorHMSDataset, POSITIONS
from hms_feature_utils import blend_patient_probabilities, build_position_distribution, slugify_label
from model_hms import HMSLiteModel
from paths import FINAL_FEATURE_CSV, FINAL_HMS_CHECKPOINT, TRAIN_CSV, TRAIN_WAV_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate patient-level HMS features.")
    parser.add_argument("--checkpoint-path", type=str, default=str(FINAL_HMS_CHECKPOINT))
    parser.add_argument("--output-path", type=str, default=str(FINAL_FEATURE_CSV))
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    return parser.parse_args()


def mean_tensor_or_zeros(tensors, size):
    if tensors:
        return torch.stack(tensors, dim=0).mean(dim=0)
    return torch.zeros(size, dtype=torch.float32)


def std_tensor_or_zeros(tensors, size):
    if tensors:
        return torch.stack(tensors, dim=0).std(dim=0, unbiased=False)
    return torch.zeros(size, dtype=torch.float32)


def max_tensor_or_zeros(tensors, size):
    if tensors:
        return torch.stack(tensors, dim=0).max(dim=0).values
    return torch.zeros(size, dtype=torch.float32)


def build_prob_columns(label_map, prefix):
    columns = []
    for label_name, _ in sorted(label_map.items(), key=lambda item: item[1]):
        columns.append((label_name, f"{prefix}_{slugify_label(label_name)}"))
    return columns


def topk_mean(values, k=3):
    if not values:
        return 0.0
    tensor_values = torch.tensor(values, dtype=torch.float32)
    topk = min(k, tensor_values.numel())
    return float(torch.topk(tensor_values, k=topk).values.mean().item())


def mean_value(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def std_value(values):
    if len(values) <= 1:
        return 0.0
    tensor_values = torch.tensor(values, dtype=torch.float32)
    return float(tensor_values.std(unbiased=False).item())


def frac_ge(values, threshold):
    if not values:
        return 0.0
    count = sum(1 for value in values if value >= threshold)
    return float(count / len(values))


def load_decision_weights(checkpoint_path):
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.exists():
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        weights = metadata.get("decision_weights", [1.0, 1.0, 1.0])
        if len(weights) != 3:
            raise ValueError("decision_weights must have length 3.")
        return torch.tensor(weights, dtype=torch.float32)
    except Exception as exc:
        print(f"Warning: failed to load decision metadata from {metadata_path}: {exc}")
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)


def calibrate_prob_tensor(prob_tensor, decision_weights):
    local_weights = decision_weights.to(prob_tensor.device).view(1, -1)
    calibrated = prob_tensor * local_weights
    calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return calibrated


def main():
    args = parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    csv_path = Path(args.csv_path).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_path = Path(args.output_path).resolve()

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
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the HMS model first or override --checkpoint-path."
        )

    print("Feature source CSV:", csv_path)
    print("Feature source WAV dir:", wav_dir)
    print("Checkpoint:", checkpoint_path)
    print("Feature output:", output_path)

    dataset = CirCorHMSDataset(
        csv_path=csv_path,
        wav_dir=wav_dir,
        sr=2000,
        segment_duration=3.0,
        segment_hop=2.0,
        n_mels=64,
    )

    model = HMSLiteModel(
        num_timing_classes=dataset.num_timing_classes,
        num_grade_classes=dataset.num_grade_classes,
        num_shape_classes=dataset.num_shape_classes,
        branch_dim=64,
        embedding_dim=128,
        murmur_classes=3,
    ).to(device)

    decision_weights = load_decision_weights(checkpoint_path)
    print("Loaded decision weights:", [round(float(x), 4) for x in decision_weights.tolist()])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint, strict=False)
    if load_result.missing_keys:
        print("Missing keys while loading checkpoint:")
        print(load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys while loading checkpoint:")
        print(load_result.unexpected_keys)

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    patient_murmur_probs = defaultdict(list)
    patient_timing_probs = defaultdict(list)
    patient_grade_probs = defaultdict(list)
    patient_shape_probs = defaultdict(list)
    patient_embeddings = defaultdict(list)
    patient_position_present_scores = defaultdict(lambda: defaultdict(list))
    patient_calib_position_present_scores = defaultdict(lambda: defaultdict(list))
    patient_segment_entropy = defaultdict(list)
    patient_segment_margin = defaultdict(list)
    patient_segment_present = defaultdict(list)
    patient_segment_absent = defaultdict(list)
    patient_segment_unknown = defaultdict(list)
    patient_calib_murmur_probs = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x1 = batch["x_scale1"].to(device)
            x2 = batch["x_scale2"].to(device)
            x3 = batch["x_scale3"].to(device)
            position_index = batch["position_index"].to(device)

            patient_ids = batch["patient_id"]
            positions = batch["position"]

            murmur_logits, timing_logits, grade_logits, shape_logits, embeddings = model(
                x1,
                x2,
                x3,
                position_index=position_index,
                return_embedding=True,
            )

            murmur_probs = torch.softmax(murmur_logits, dim=1).cpu()
            calib_murmur_probs = calibrate_prob_tensor(murmur_probs, decision_weights).cpu()
            timing_probs = torch.softmax(timing_logits, dim=1).cpu() if timing_logits is not None else None
            grade_probs = torch.softmax(grade_logits, dim=1).cpu() if grade_logits is not None else None
            shape_probs = torch.softmax(shape_logits, dim=1).cpu() if shape_logits is not None else None
            embeddings = embeddings.cpu()

            for idx, patient_id in enumerate(patient_ids):
                patient_murmur_probs[patient_id].append(murmur_probs[idx])
                patient_calib_murmur_probs[patient_id].append(calib_murmur_probs[idx])
                patient_embeddings[patient_id].append(embeddings[idx])
                patient_segment_present[patient_id].append(float(murmur_probs[idx][1].item()))
                patient_segment_absent[patient_id].append(float(murmur_probs[idx][0].item()))
                patient_segment_unknown[patient_id].append(float(murmur_probs[idx][2].item()))
                patient_segment_margin[patient_id].append(float((murmur_probs[idx][1] - murmur_probs[idx][0]).item()))

                entropy = -(murmur_probs[idx] * (murmur_probs[idx].clamp_min(1e-8).log())).sum()
                patient_segment_entropy[patient_id].append(float(entropy.item()))

                if timing_probs is not None:
                    patient_timing_probs[patient_id].append(timing_probs[idx])
                if grade_probs is not None:
                    patient_grade_probs[patient_id].append(grade_probs[idx])
                if shape_probs is not None:
                    patient_shape_probs[patient_id].append(shape_probs[idx])

                position_name = positions[idx]
                patient_position_present_scores[patient_id][position_name].append(float(murmur_probs[idx][1].item()))
                patient_calib_position_present_scores[patient_id][position_name].append(float(calib_murmur_probs[idx][1].item()))

    timing_columns = build_prob_columns(dataset.timing_map, "timing_prob")
    grade_columns = build_prob_columns(dataset.grade_map, "grade_prob")
    shape_columns = build_prob_columns(dataset.shape_map, "shape_prob")

    rows = []
    for patient_id in sorted(patient_murmur_probs.keys()):
        murmur_stack = torch.stack(patient_murmur_probs[patient_id], dim=0)
        murmur_mean = murmur_stack.mean(dim=0)
        murmur_max = murmur_stack.max(dim=0).values
        murmur_std = murmur_stack.std(dim=0, unbiased=False)
        murmur_blended = blend_patient_probabilities(patient_murmur_probs[patient_id])
        calib_murmur_stack = torch.stack(patient_calib_murmur_probs[patient_id], dim=0)
        calib_murmur_mean = calib_murmur_stack.mean(dim=0)
        calib_murmur_max = calib_murmur_stack.max(dim=0).values
        calib_murmur_std = calib_murmur_stack.std(dim=0, unbiased=False)
        calib_murmur_blended = blend_patient_probabilities(patient_calib_murmur_probs[patient_id])

        timing_mean = mean_tensor_or_zeros(patient_timing_probs[patient_id], dataset.num_timing_classes)
        timing_max = max_tensor_or_zeros(patient_timing_probs[patient_id], dataset.num_timing_classes)
        grade_mean = mean_tensor_or_zeros(patient_grade_probs[patient_id], dataset.num_grade_classes)
        grade_max = max_tensor_or_zeros(patient_grade_probs[patient_id], dataset.num_grade_classes)
        shape_mean = mean_tensor_or_zeros(patient_shape_probs[patient_id], dataset.num_shape_classes)
        shape_max = max_tensor_or_zeros(patient_shape_probs[patient_id], dataset.num_shape_classes)
        embedding_mean = torch.stack(patient_embeddings[patient_id], dim=0).mean(dim=0)

        position_scores = {}
        calib_position_scores = {}
        for position_name in POSITIONS:
            values = patient_position_present_scores[patient_id].get(position_name, [])
            position_scores[position_name] = max(values) if values else 0.0
            calib_values = patient_calib_position_present_scores[patient_id].get(position_name, [])
            calib_position_scores[position_name] = max(calib_values) if calib_values else 0.0
        position_distribution = build_position_distribution(position_scores)
        calib_position_distribution = build_position_distribution(calib_position_scores)

        row = {
            "Patient ID": patient_id,
            "prob_absent": float(murmur_mean[0].item()),
            "prob_present": float(murmur_mean[1].item()),
            "prob_unknown": float(murmur_mean[2].item()),
            "calib_prob_absent": float(calib_murmur_mean[0].item()),
            "calib_prob_present": float(calib_murmur_mean[1].item()),
            "calib_prob_unknown": float(calib_murmur_mean[2].item()),
            "max_prob_absent": float(murmur_max[0].item()),
            "max_prob_present": float(murmur_max[1].item()),
            "max_prob_unknown": float(murmur_max[2].item()),
            "calib_max_prob_absent": float(calib_murmur_max[0].item()),
            "calib_max_prob_present": float(calib_murmur_max[1].item()),
            "calib_max_prob_unknown": float(calib_murmur_max[2].item()),
            "std_prob_absent": float(murmur_std[0].item()),
            "std_prob_present": float(murmur_std[1].item()),
            "std_prob_unknown": float(murmur_std[2].item()),
            "calib_std_prob_absent": float(calib_murmur_std[0].item()),
            "calib_std_prob_present": float(calib_murmur_std[1].item()),
            "calib_std_prob_unknown": float(calib_murmur_std[2].item()),
            "num_segments": int(murmur_stack.size(0)),
            "murmur_prob_overall": float(murmur_blended[1].item()),
            "calib_murmur_prob_overall": float(calib_murmur_blended[1].item()),
            "segment_present_top3_mean": topk_mean(patient_segment_present[patient_id], k=3),
            "segment_absent_top3_mean": topk_mean(patient_segment_absent[patient_id], k=3),
            "segment_unknown_top3_mean": topk_mean(patient_segment_unknown[patient_id], k=3),
            "segment_present_frac_ge_040": frac_ge(patient_segment_present[patient_id], 0.40),
            "segment_present_frac_ge_050": frac_ge(patient_segment_present[patient_id], 0.50),
            "segment_present_frac_ge_060": frac_ge(patient_segment_present[patient_id], 0.60),
            "segment_present_frac_ge_070": frac_ge(patient_segment_present[patient_id], 0.70),
            "segment_unknown_frac_ge_030": frac_ge(patient_segment_unknown[patient_id], 0.30),
            "segment_margin_mean": mean_value(patient_segment_margin[patient_id]),
            "segment_margin_std": std_value(patient_segment_margin[patient_id]),
            "segment_margin_top3_mean": topk_mean(patient_segment_margin[patient_id], k=3),
            "segment_entropy_mean": mean_value(patient_segment_entropy[patient_id]),
            "segment_entropy_std": std_value(patient_segment_entropy[patient_id]),
            "segment_entropy_top3_mean": topk_mean(patient_segment_entropy[patient_id], k=3),
        }

        for position_name in POSITIONS:
            position_values = patient_position_present_scores[patient_id].get(position_name, [])
            row[f"murmur_prob_{position_name}"] = float(position_distribution[position_name])
            row[f"calib_murmur_prob_{position_name}"] = float(calib_position_distribution[position_name])
            row[f"position_present_max_{position_name}"] = float(position_scores[position_name])
            row[f"position_present_mean_{position_name}"] = mean_value(position_values)
            row[f"position_present_std_{position_name}"] = std_value(position_values)
            row[f"position_present_top2_mean_{position_name}"] = topk_mean(position_values, k=2)

        for class_index, (_, column_name) in enumerate(timing_columns):
            row[column_name] = float(timing_mean[class_index].item())
            row[column_name.replace("timing_prob_", "timing_max_")] = float(timing_max[class_index].item())

        for class_index, (_, column_name) in enumerate(grade_columns):
            row[column_name] = float(grade_mean[class_index].item())
            row[column_name.replace("grade_prob_", "grade_max_")] = float(grade_max[class_index].item())

        for class_index, (_, column_name) in enumerate(shape_columns):
            row[column_name] = float(shape_mean[class_index].item())
            row[column_name.replace("shape_prob_", "shape_max_")] = float(shape_max[class_index].item())

        for emb_index, emb_value in enumerate(embedding_mean.tolist()):
            row[f"embedding_{emb_index}"] = float(emb_value)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved features to: {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()
