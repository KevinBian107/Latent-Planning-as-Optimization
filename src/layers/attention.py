import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out

class CrossAttention(nn.Module):
    def __init__(self, h_dim, n_heads, dropout=0.1):
        super().__init__()
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        assert self.head_dim * n_heads == h_dim, "hidden dim must be divisible by n_heads"

        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Linear(h_dim, h_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_bias=None, attention_mask=None):
        """
        query: (batch_size, tgt_len, h_dim)
        key: (batch_size, src_len, h_dim)
        value: (batch_size, src_len, h_dim)
        attn_bias: (batch_size, n_heads, tgt_len, src_len) or None
            - float tensor: positive or negative bias to add to attention logits
        attention_mask: (batch_size, tgt_len, src_len) or None
            - bool tensor: True=mask，False=normal
        """
        B, tgt_len, _ = query.size()
        B, src_len, _ = key.size()

        # Linear projections
        q = self.q_proj(query).view(B, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, tgt_len, head_dim)
        k = self.k_proj(key).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2)    # (B, n_heads, src_len, head_dim)
        v = self.v_proj(value).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (B, n_heads, tgt_len, src_len)
        attn_logits = attn_logits / (self.head_dim ** 0.5)

        # --- 加 bias ---
        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias  

        # --- 加 mask ---
        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(attention_mask.unsqueeze(1), float('-inf'))  # bool mask, True=mask掉

        # --- softmax ---
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # attention output
        attn_output = torch.matmul(attn_probs, v)  # (B, n_heads, tgt_len, head_dim)

        # reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, tgt_len, self.h_dim)

        return self.out_proj(attn_output)

