import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLPGenerator(nn.Module):
    """Generator that maps z0 to z conditioned on alpha using FiLM modulation"""
    def __init__(self, z_dim: int, alpha_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.alpha_dim = alpha_dim

        # MLP that maps z0 → z, conditioned on alpha via FiLM
        self.fc1 = nn.Linear(z_dim, hidden_dim)  # Input: [B, z_dim], Output: [B, hidden_dim]
        self.fc2 = nn.Linear(hidden_dim, z_dim)  # Input: [B, hidden_dim], Output: [B, z_dim]

        # Input: [B, alpha_dim], Output: [B, 2*hidden_dim]
        self.film1 = nn.Linear(alpha_dim, 2 * hidden_dim)
        # Input: [B, alpha_dim], Output: [B, 2*z_dim]
        self.film2 = nn.Linear(alpha_dim, 2 * z_dim)

        self.activation = nn.ReLU()

    def apply_film(self, x: torch.Tensor, film: nn.Linear, alpha: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM (Feature-wise Linear Modulation),
        we take in alpha (tensor) that shifts the weight distribution of an existing MLP
        
        Args:
            x: Feature tensor [B, dim] or [B, seq_len, dim]
            film: Linear layer for FiLM parameters
            alpha: Conditioning tensor [B, alpha_dim]
            
        Returns:
            Modulated features (same shape as x)
        """
        # alpha give scale and shift parameters
        scale_shift = film(alpha)  # [B, 2*dim]
        if x.dim() == 3:
            scale_shift = scale_shift.unsqueeze(1)  # [B, 1, 2*dim]
            
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each: [B, dim] or [B, 1, dim]
        
        # modulation: out = x * (1 + scale) + shift
        return x * (1 + scale) + shift

    def forward(self, z0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of generator
        
        Args:
            z0: Initial latent code [B, z_dim]
            alpha: Conditioning parameter [B, alpha_dim]
            
        Returns:
            Generated latent code z [B, z_dim]
        """
        x = self.fc1(z0)  # [B, hidden_dim]
        x = self.apply_film(x, self.film1, alpha)  # [B, hidden_dim]
        x = self.activation(x)  # [B, hidden_dim]

        x = self.fc2(x)  # [B, z_dim]
        x = self.apply_film(x, self.film2, alpha)  # [B, z_dim]
        return x  # [B, z_dim]