import torch
import minari
from collections import deque
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = torch.load("results/weights/lpt_halfcheetah.pt", map_location="cpu",weights_only=False)  # or to('cuda')
model.to(device)
model.device = device
model.eval()

dataset = minari.load_dataset('mujoco/halfcheetah/expert-v0')
env = dataset.recover_environment(render_mode="human")
obs = env.reset()[0]

context_len = 15
state_dim = obs.shape[0]
act_dim = model.act_dim  # assume this exists, otherwise hardcode

# Deques for context window
state_buffer = deque(maxlen=context_len)
action_buffer = deque(maxlen=context_len)
timestep_buffer = deque(maxlen=context_len)

# Init with padding
for t in range(context_len-1):
    state_buffer.append(np.zeros(state_dim))
    action_buffer.append(np.zeros(act_dim))
    timestep_buffer.append(t)

state_buffer.append(obs)
action_buffer.append(np.zeros(act_dim))
timestep_buffer.append(context_len)

STEP = 1000
for t in range(STEP):
    # Assemble model input
    states = torch.tensor([list(state_buffer)], dtype=torch.float32).to(device = device)
    actions = torch.tensor([list(action_buffer)], dtype=torch.float32).to(device = device)
    rewards = torch.tensor([100], dtype=torch.float32).reshape(1,1).to(device = device)
    timesteps = torch.tensor([list(timestep_buffer)], dtype=torch.long).to(device = device)

    # Forward pass
    pred_action, _, _ = model(states, actions, timesteps, rewards, batch_inds = torch.tensor([0]).to(device)) #batch_inds can be any tesnor actually as it won't be used in eval
    action = pred_action.squeeze().detach().cpu().numpy()  # last token prediction

    # Step environment
    next_obs, reward, done, truncated, info = env.step(action)
    env.render()

    # Update context
    state_buffer.append(next_obs)
    action_buffer.append(action)
    timestep_buffer.append(t + context_len)

    if done or truncated:
        obs = env.reset()[0]
        state_buffer.clear()
        action_buffer.clear()
        timestep_buffer.clear()
        for t in range(context_len-1):
            state_buffer.append(np.zeros(state_dim))
            action_buffer.append(np.zeros(act_dim))
            timestep_buffer.append(t)

        state_buffer.append(obs)
        action_buffer.append(np.zeros(act_dim))
        timestep_buffer.append(context_len)