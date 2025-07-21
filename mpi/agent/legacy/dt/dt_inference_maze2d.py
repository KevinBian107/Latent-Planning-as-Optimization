import torch
import minari
from collections import deque
import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
print(os.getcwd())
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = torch.load("results/weights/dt_maze2d_dense_umaze.pt", weights_only=False)
model.to(device)
model.device = device
model.eval()

dataset = minari.load_dataset('D4RL/pointmaze/large-dense-v2', download=True)
env = dataset.recover_environment(render_mode="human", eval_env=True)


obs_dict, _ = env.reset()
obs        = obs_dict['observation']
desired_g  = obs_dict['desired_goal']
achieved_g = obs_dict['achieved_goal']

full_state = np.concatenate([obs, desired_g, achieved_g], axis=0)
state_dim  = full_state.shape[0]
action_dim = env.action_space.shape[0]

max_len      = 150
states       = deque([], maxlen=max_len)
actions      = deque([], maxlen=max_len)
rewards      = deque([], maxlen=max_len)
timesteps    = deque([], maxlen=max_len)
rtgs         = deque([], maxlen=max_len)

total_reward = 0.0
timestep     = 0

# initial target return (can tune this)
target_return = 80.0
for _ in range(max_len):
    states.append(torch.tensor(full_state, dtype=torch.float32))
    actions.append(torch.zeros(action_dim))
    rewards.append(0.0)
    timesteps.append(0)
    rtgs.append(torch.tensor([target_return - total_reward], dtype=torch.float32))

def pad_seq(seq, length, dim):
    """pad/truncate a sequence of vectors to (length, dim)"""
    data = torch.stack(list(seq))  # (len(seq), dim)
    if len(seq) < length:
        pad = torch.zeros((length - len(seq), dim), dtype=data.dtype)
        return torch.cat([pad, data], dim=0)
    else:
        return data[-length:]

def pad_scalar(seq, length):
    """pad/truncate a sequence of scalars to (length,)"""
    data = torch.tensor(list(seq), dtype=torch.long)
    if len(data) < length:
        pad = torch.zeros(length - len(data), dtype=torch.long)
        return torch.cat([pad, data], dim=0)
    else:
        return data[-length:]

for step in range(5000):
    # build batch tensors
    states_tensor   = pad_seq(states,   max_len, state_dim).unsqueeze(0).to(device)
    actions_tensor  = pad_seq(actions,  max_len, action_dim).unsqueeze(0).to(device)
    rtgs_tensor     = pad_seq(rtgs,     max_len, 1).unsqueeze(0).to(device)
    timesteps_tensor= pad_scalar(timesteps, max_len).unsqueeze(0).to(device)

    with torch.no_grad():
        _, action_preds, _ = model(
            timesteps=timesteps_tensor,
            states=states_tensor,
            actions=actions_tensor,
            returns_to_go=rtgs_tensor
        )
        action = action_preds[0, -1].cpu().numpy()

    # step environment
    print(action)
    time.sleep(1)
    obs_dict, reward, done, _, _ = env.step(action)
    obs        = obs_dict['observation']
    desired_g  = obs_dict['desired_goal']
    achieved_g = obs_dict['achieved_goal']
    full_state = np.concatenate([obs, desired_g, achieved_g], axis=0)

    total_reward += reward
    print(f"Step {step}, reward={reward:.3f}, total={total_reward:.3f}")

    # slide the buffer
    actions.popleft()
    states.popleft()
    rewards.popleft()
    rtgs.popleft()
    timesteps.popleft()

    actions.append(torch.tensor(action, dtype=torch.float32))
    states.append(torch.tensor(full_state, dtype=torch.float32))
    rewards.append(reward)
    # update return-to-go = (target_return - total_reward)
    rtgs.append(torch.tensor([target_return - total_reward], dtype=torch.float32))
    timesteps.append(timestep)
    timestep += 1

    if done:
        break
