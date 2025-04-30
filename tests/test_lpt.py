import os
import sys
import torch


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from src.models.LPT import LatentPlannerModel

def test_latent_planner_model():
    batch_size = 4
    context_len = 8
    state_dim = 17
    act_dim = 6

    # Instantiate the model
    model = LatentPlannerModel(
        state_dim=state_dim,
        act_dim=act_dim,
        context_len=context_len,
        h_dim=32,
        n_blocks=2,
        n_heads=2,
        drop_p=0.1,
        n_latent=4
    )

    # Generate random inputs
    states = torch.randn(batch_size, context_len, state_dim)
    actions = torch.randn(batch_size, context_len, act_dim)
    timesteps = torch.randint(0, 1000, (batch_size, context_len))
    rewards = torch.randn(batch_size)
    batch_inds = torch.randint(0, 1000, (batch_size,))

    # Forward pass
    pred_action, pred_state, pred_reward = model(states, actions, timesteps, rewards, batch_inds)

    # Check shapes
    assert pred_action.shape == (batch_size, act_dim), f"Expected pred_action shape {(batch_size, act_dim)}, got {pred_action.shape}"
    assert pred_state.shape == (batch_size, state_dim), f"Expected pred_state shape {(batch_size, state_dim)}, got {pred_state.shape}"
    assert pred_reward.shape == (batch_size,), f"Expected pred_reward shape {(batch_size,)}, got {pred_reward.shape}"

    print("Test passed: LatentPlannerModel outputs have correct shapes.")

if __name__ == "__main__":
    test_latent_planner_model()