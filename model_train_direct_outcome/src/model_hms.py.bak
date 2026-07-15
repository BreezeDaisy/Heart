import torch
import torch.nn as nn


class ScaleEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class HMSLiteModel(nn.Module):
    def __init__(
        self,
        num_timing_classes,
        num_grade_classes,
        num_shape_classes=0,
        branch_dim=64,
        embedding_dim=128,
        murmur_classes=3,
        num_positions=4,
        position_dim=16,
    ):
        super().__init__()

        self.scale1_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale2_encoder = ScaleEncoder(out_dim=branch_dim)
        self.scale3_encoder = ScaleEncoder(out_dim=branch_dim)
        self.position_embedding = nn.Embedding(num_positions, position_dim)

        fused_dim = branch_dim * 3 + position_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.murmur_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, murmur_classes),
        )

        self.timing_head = nn.Linear(embedding_dim, num_timing_classes) if num_timing_classes > 0 else None
        self.grade_head = nn.Linear(embedding_dim, num_grade_classes) if num_grade_classes > 0 else None
        self.shape_head = nn.Linear(embedding_dim, num_shape_classes) if num_shape_classes > 0 else None

    def forward(self, x_scale1, x_scale2, x_scale3, position_index=None, return_embedding=False):
        f1 = self.scale1_encoder(x_scale1)
        f2 = self.scale2_encoder(x_scale2)
        f3 = self.scale3_encoder(x_scale3)

        if position_index is None:
            position_index = torch.zeros(
                x_scale1.size(0),
                dtype=torch.long,
                device=x_scale1.device,
            )
        position_feature = self.position_embedding(position_index.long())

        fused = torch.cat([f1, f2, f3, position_feature], dim=1)
        fused = self.fusion(fused)
        embedding = self.embedding_head(fused)

        murmur_logits = self.murmur_head(embedding)
        timing_logits = self.timing_head(embedding) if self.timing_head is not None else None
        grade_logits = self.grade_head(embedding) if self.grade_head is not None else None
        shape_logits = self.shape_head(embedding) if self.shape_head is not None else None

        if return_embedding:
            return murmur_logits, timing_logits, grade_logits, shape_logits, embedding

        return murmur_logits, timing_logits, grade_logits, shape_logits
