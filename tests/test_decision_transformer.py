import os
import sys
import torch
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from src.models.decision_transformer import DecisionTransformer
state_dim = 17    # eg: in HalfCheetah, state_dim = 17
act_dim = 6       # dim of action space
n_blocks = 3      # transformer block num
h_dim = 128       # hidden size
context_len = 20  # previous 20 steps
n_heads = 8       # multihead attention
drop_p = 0.1
try:
    model = DecisionTransformer(state_dim = state_dim,
                                act_dim = act_dim,
                                n_blocks = n_blocks,
                                h_dim = h_dim,
                                context_len = context_len,
                                n_heads = n_heads,
                                drop_p = drop_p
                                )
except Exception as e:
    print("")
    raise e
print("="*30)
print("Decision Transformer Structure")
print("="*30)
print(model)

batch_size = 4  # batch size

timesteps = torch.randint(0, 4096, (batch_size, context_len))  # idx of each timestep
states = torch.randn(batch_size, context_len, state_dim)       # random state
actions = torch.randn(batch_size, context_len, act_dim)        # random actions
returns_to_go = torch.randn(batch_size, context_len, 1)        # random return-to-go

# fit in model
state_preds, action_preds, return_preds = model(
    timesteps=timesteps,
    states=states,
    actions=actions,
    returns_to_go=returns_to_go
)

print("=" * 40)
print("Test Input")
print("=" * 40)
print(f"{'Name':<20}{'Shape'}")
print("-" * 40)
print(f"{'timesteps':<20}{timesteps.shape}")
print(f"{'states':<20}{states.shape}")
print(f"{'actions':<20}{actions.shape}")
print(f"{'returns_to_go':<20}{returns_to_go.shape}")

print("\n" + "=" * 40)
print("Model Output")
print("=" * 40)
print(f"{'Name':<20}{'Shape'}")
print("-" * 40)
print(f"{'state_preds':<20}{state_preds.shape}")
print(f"{'action_preds':<20}{action_preds.shape}")
print(f"{'return_preds':<20}{return_preds.shape}")
print("=" * 40)

assert state_preds.shape == states.shape, \
    f"Shape mismatch: state_preds {state_preds.shape} vs states {states.shape}"

assert action_preds.shape == actions.shape, \
    f"Shape mismatch: action_preds {action_preds.shape} vs actions {actions.shape}"

assert return_preds.shape == returns_to_go.shape, \
    f"Shape mismatch: return_preds {return_preds.shape} vs returns_to_go {returns_to_go.shape}"

print("\nAll output shapes match the input shapes! ✅")
