import re

import torch

from dataset_hms import POSITIONS


def slugify_label(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def blend_patient_probabilities(prob_list):
    if len(prob_list) == 1:
        return prob_list[0]

    stacked = torch.stack(prob_list, dim=0)
    mean_prob = stacked.mean(dim=0)

    abnormal_score = stacked[:, 1] + 0.25 * stacked[:, 2]
    topk = min(3, stacked.size(0))
    topk_indices = torch.topk(abnormal_score, k=topk).indices
    topk_mean = stacked[topk_indices].mean(dim=0)
    max_prob = stacked.max(dim=0).values

    blended = 0.75 * mean_prob + 0.15 * topk_mean + 0.10 * max_prob
    return blended / blended.sum().clamp_min(1e-8)


def build_position_distribution(position_scores):
    total = sum(position_scores.values())
    if total <= 1e-8:
        uniform = 1.0 / len(POSITIONS)
        return {pos: uniform for pos in POSITIONS}
    return {pos: position_scores[pos] / total for pos in POSITIONS}
