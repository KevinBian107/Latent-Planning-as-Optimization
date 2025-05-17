import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ANILGenerator(nn.Module):
    def __init__(self, z_dim: int, alpha_feat_dim: int, alpha_head_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Feature extractor (frozen)
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.activation = nn.ReLU()

        # FiLM for feature extractor (large, frozen)
        self.film1 = nn.Linear(alpha_feat_dim, 2 * hidden_dim)

        # Task-specific head (small, trainable)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.film2 = nn.Linear(alpha_head_dim, 2 * z_dim)

        self.freeze_feature_extractor()

    def freeze_feature_extractor(self):
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.film1.parameters():
            param.requires_grad = False

    def extract_feat(self, z0, requires_grad=False):
        if requires_grad:
            return self.activation(self.fc1(z0))
        else:
            with torch.no_grad():
                return self.activation(self.fc1(z0))


    def apply_film(self, x, film_layer, alpha):
        scale_shift = film_layer(alpha)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1 + scale) + shift


    def forward(self, z0, alpha_feat, alpha_head, requires_grad=False):
        x = self.extract_feat(z0, requires_grad=requires_grad)
        x = self.apply_film(x, self.film1, alpha_feat)
        x = self.fc2(x)
        x = self.apply_film(x, self.film2, alpha_head)
        return x

