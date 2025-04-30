
import torch
import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from src.models.conditional_decision_transformer import ConditionalDecisionTransformer
batch_size = 4
context_len = 6
state_dim = 17
act_dim = 4
h_dim = 64
n_blocks = 2
n_heads = 2
drop_p = 0.1
max_timestep = 100

model = ConditionalDecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=n_blocks,
    h_dim=h_dim,
    context_len=context_len,
    n_heads=n_heads,
    drop_p=drop_p,
    max_timestep=max_timestep,
    action_tanh=True
)

timesteps = torch.randint(0, max_timestep, (batch_size, context_len))
states = torch.randn(batch_size, context_len, state_dim)
actions = torch.randn(batch_size, context_len, act_dim)
z_latent = torch.randn(batch_size, context_len, h_dim)  # cross-attention memory

pred_action, pred_state = model(timesteps, states, actions, z_latent)

assert pred_action.shape == (batch_size, act_dim), f"Unexpected pred_action shape: {pred_action.shape}"
assert pred_state.shape == (batch_size, state_dim), f"Unexpected pred_state shape: {pred_state.shape}"
print("Test passed: ConditionalDecisionTransformer outputs correct shapes.")