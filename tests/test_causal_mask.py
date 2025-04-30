import torch
import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
import math
from src.layers.attention import MaskedCausalAttention
def test_masked_causal_attention():
    h_dim = 8
    n_heads = 2
    max_T = 5
    drop_p = 0.0  
    B = 1  
    T = 5  

    attention_layer = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
    
    x = torch.randn(B, T, h_dim)

    B, T, C = x.shape
    N, D = n_heads, C // n_heads

    q = attention_layer.q_net(x).view(B, T, N, D).transpose(1,2)
    k = attention_layer.k_net(x).view(B, T, N, D).transpose(1,2)
    v = attention_layer.v_net(x).view(B, T, N, D).transpose(1,2)

    weights = q @ k.transpose(2,3) / math.sqrt(D)
    weights = weights.masked_fill(attention_layer.mask[...,:T,:T] == 0, float('-inf'))
    normalized_weights = torch.softmax(weights, dim=-1)  

    print("Attention weights (after masking and softmax):")
    print(normalized_weights[0, 0]) 

    # for each timestep, it should not see the future.
    for i in range(T):
        # the attention of future steps shuld close to 0
        future_attention = normalized_weights[0, 0, i, i+1:]
        assert torch.all(future_attention < 1e-4), f"Future attention at timestep {i} is not zero!"

    print("Masked causal attention passed ✅")

test_masked_causal_attention()
