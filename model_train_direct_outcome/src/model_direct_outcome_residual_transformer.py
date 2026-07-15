import torch
import torch.nn as nn

from model_hms import ScaleEncoder


class DirectOutcomeMultiTaskModel(nn.Module):
    """Position-aware patient-level model used before the residual-transformer experiment.

    This file is checkpoint-compatible with the position-aware lightweight CNN
    checkpoints such as direct_outcome_gpu_position_aware_fp010_v1.pth.
    """

    def __init__(
        self,
        clinical_dim,
        num_timing_classes,
        num_grade_classes,
        num_shape_classes,
        branch_dim=64,
        segment_embedding_dim=128,
        patient_embedding_dim=160,
        clinical_embedding_dim=32,
        num_positions=4,
        position_dim=16,
        outcome_classes=2,
        murmur_classes=3,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.scale1_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale2_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale3_encoder = ScaleEncoder(out_dim=branch_dim)
        self.position_embedding = nn.Embedding(num_positions, position_dim)

        fused_segment_dim = branch_dim * 3 + position_dim
        self.segment_head = nn.Sequential(
            nn.Linear(fused_segment_dim, 192),
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

        self.clinical_head = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, clinical_embedding_dim),
            nn.ReLU(inplace=True),
        )

        patient_input_dim = (segment_embedding_dim * 2 * num_positions) + num_positions + clinical_embedding_dim
        self.patient_head = nn.Sequential(
            nn.Linear(patient_input_dim, patient_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(patient_embedding_dim, patient_embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.outcome_head = nn.Linear(patient_embedding_dim, outcome_classes)
        self.murmur_head = nn.Linear(patient_embedding_dim, murmur_classes)
        self.timing_head = nn.Linear(patient_embedding_dim, num_timing_classes) if num_timing_classes > 0 else None
        self.grade_head = nn.Linear(patient_embedding_dim, num_grade_classes) if num_grade_classes > 0 else None
        self.shape_head = nn.Linear(patient_embedding_dim, num_shape_classes) if num_shape_classes > 0 else None

    def _encode_segments(self, x_scale1, x_scale2, x_scale3, position_index):
        batch_size, num_segments = x_scale1.shape[:2]
        flat_x1 = x_scale1.reshape(batch_size * num_segments, *x_scale1.shape[2:])
        flat_x2 = x_scale2.reshape(batch_size * num_segments, *x_scale2.shape[2:])
        flat_x3 = x_scale3.reshape(batch_size * num_segments, *x_scale3.shape[2:])
        flat_positions = position_index.reshape(batch_size * num_segments)

        f1 = self.scale1_encoder(flat_x1)
        f2 = self.scale2_encoder(flat_x2)
        f3 = self.scale3_encoder(flat_x3)
        pos_features = self.position_embedding(flat_positions.long())
        segment_features = torch.cat([f1, f2, f3, pos_features], dim=1)
        segment_embeddings = self.segment_head(segment_features)
        return segment_embeddings.view(batch_size, num_segments, -1)

    def _pool_by_position(self, segment_embeddings, position_index, mask):
        attention_logits = self.attention(segment_embeddings).squeeze(-1)
        position_features = []
        position_presence = []
        position_attention_weights = torch.zeros_like(attention_logits)

        for pos_idx in range(self.num_positions):
            pos_mask = (position_index.long() == pos_idx) & (mask > 0.0)
            pos_mask_float = pos_mask.float()
            has_position = pos_mask_float.sum(dim=1, keepdim=True) > 0.0
            position_presence.append(has_position.float())

            pos_logits = attention_logits.masked_fill(~pos_mask, -1e4)
            pos_weights = torch.softmax(pos_logits, dim=1) * pos_mask_float
            pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            position_attention_weights = position_attention_weights + pos_weights
            attentive_pool = torch.sum(segment_embeddings * pos_weights.unsqueeze(-1), dim=1)

            max_pool = segment_embeddings.masked_fill(~pos_mask.unsqueeze(-1), -1e4).max(dim=1).values
            max_pool = torch.where(max_pool < -1e3, torch.zeros_like(max_pool), max_pool)
            position_features.extend([attentive_pool, max_pool])

        return (
            torch.cat(position_features, dim=1),
            torch.cat(position_presence, dim=1),
            position_attention_weights,
        )

    def forward(
        self,
        x_scale1,
        x_scale2,
        x_scale3,
        position_index,
        segment_mask,
        clinical,
        return_embedding=False,
    ):
        segment_embeddings = self._encode_segments(x_scale1, x_scale2, x_scale3, position_index)
        mask = segment_mask.float().clamp(0.0, 1.0)
        position_pool, position_presence, attention_weights = self._pool_by_position(
            segment_embeddings,
            position_index,
            mask,
        )

        clinical_embedding = self.clinical_head(clinical.float())
        patient_features = torch.cat([position_pool, position_presence, clinical_embedding], dim=1)
        patient_embedding = self.patient_head(patient_features)

        outcome_logits = self.outcome_head(patient_embedding)
        murmur_logits = self.murmur_head(patient_embedding)
        timing_logits = self.timing_head(patient_embedding) if self.timing_head is not None else None
        grade_logits = self.grade_head(patient_embedding) if self.grade_head is not None else None
        shape_logits = self.shape_head(patient_embedding) if self.shape_head is not None else None

        if return_embedding:
            return {
                "outcome_logits": outcome_logits,
                "murmur_logits": murmur_logits,
                "timing_logits": timing_logits,
                "grade_logits": grade_logits,
                "shape_logits": shape_logits,
                "patient_embedding": patient_embedding,
                "attention_weights": attention_weights,
                "position_presence": position_presence,
            }

        return {
            "outcome_logits": outcome_logits,
            "murmur_logits": murmur_logits,
            "timing_logits": timing_logits,
            "grade_logits": grade_logits,
            "shape_logits": shape_logits,
        }
