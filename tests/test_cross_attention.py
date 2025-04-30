import os
import sys
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from src.layers.attention import CrossAttention
attn = CrossAttention(h_dim=512, n_heads=8)
attn.eval()  # 

query = torch.randn(4, 10, 512)
key = torch.randn(4, 5, 512)
value = torch.randn(4, 5, 512)

# add bias by hands
bias = torch.zeros(4, 8, 10, 5)
bias[:, :, :, :2] = 5.0  # force a high score in the 0dx,1dx keys

# mask the 4dx (5th) key
mask = torch.zeros(4, 10, 5, dtype=torch.bool)
mask[:, :, 4] = True

# --- get attention scores ---
def forward_with_attn_scores(self, query, key, value, attn_bias=None, attention_mask=None):
    B, tgt_len, _ = query.size()
    B, src_len, _ = key.size()

    q = self.q_proj(query).view(B, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(key).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(value).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2)

    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

    if attn_bias is not None:
        attn_logits = attn_logits + attn_bias

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1)
        attn_logits = attn_logits.masked_fill(attention_mask, float('-inf'))

    attn_probs = torch.softmax(attn_logits, dim=-1)

    attn_output = torch.matmul(attn_probs, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, tgt_len, self.h_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_probs  

#bind the function directly to the cross_attention
CrossAttention.forward_with_scores = forward_with_attn_scores


out, attn_probs = attn.forward_with_scores(query, key, value, attn_bias=bias, attention_mask=mask)
print(out.shape)  # (4, 10, 512)
print(attn_probs.shape)  # (4, 8, 10, 5)

#check if the bias is working
attn_focus = attn_probs[0, 0, 0]  # get 0dx sample, 0dx heads, 0dx keys
print(f"Attention distribution for first query: {attn_focus}")

assert attn_focus[0] > attn_focus[2], "Bias on key 0 not effective enough"
assert attn_focus[1] > attn_focus[2], "Bias on key 1 not effective enough"
print("✅ Bias testing passed: preferred keys get higher attention!")

#check if mask is working
assert torch.allclose(attn_focus[4], torch.tensor(0.0), atol=1e-4), "Masked key attention not close to zero!"
print("✅ Mask testing passed: masked keys have near-zero attention!")
