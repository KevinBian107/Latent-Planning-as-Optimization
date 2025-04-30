import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.attention import MaskedCausalAttention,CrossAttention
from src.util_func.unet1d_func import exists

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x
    

class CrossAttnBlock(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.self_attn = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.cross_attn = CrossAttention(h_dim, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.ln3 = nn.LayerNorm(h_dim)

    def forward(self, x_tuple):
        x, z_latent = x_tuple
        # Masked Self Attention
        x = x + self.self_attn(x)
        x = self.ln1(x)
        # Cross Attention (query: x, key/value: z_latent)
        x = x + self.cross_attn(x, z_latent, z_latent)
        x = self.ln2(x)
        # MLP
        x = x + self.mlp(x)
        x = self.ln3(x)
        return x



class BasicResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        num_groups = 8 if dim % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
    
class UnetAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch, channel, length) -> (batch, length, channel)
        x = x.permute(0, 2, 1)
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        return x.permute(0, 2, 1)