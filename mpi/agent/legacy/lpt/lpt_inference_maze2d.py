import torch
import minari
from collections import deque
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

model = torch.load("results/weights/lpt_maze2d_umaze.pt", map_location="cpu" ,weights_only=False)  # or to('cuda')
device ='cpu'

model.to(device)
model.device = device
model.eval()

# dataset = minari.load_dataset('D4RL/pointmaze/medium-dense-v2')
dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download=True)

env = dataset.recover_environment(render_mode="human")
obs = env.reset()[0]
obs = torch.cat([torch.tensor(obs["observation"]), torch.tensor(obs["achieved_goal"]), torch.tensor(obs["desired_goal"])],dim = -1).to(device)

context_len = 5
state_dim = obs.shape[0]
act_dim = model.act_dim

# Deques for context window
state_buffer = deque(maxlen=context_len)
action_buffer = deque(maxlen=context_len)
timestep_buffer = deque(maxlen=context_len)

# Init with padding
for t in range(context_len-1):
    state_buffer.append(np.zeros(state_dim))
    action_buffer.append(np.zeros(act_dim))
    timestep_buffer.append(t)

state_buffer.append(obs.cpu().numpy())
action_buffer.append(np.zeros(act_dim))
timestep_buffer.append(context_len)

STEP = 1000
total_reward = 0
for t in range(STEP):
    states = torch.tensor([list(state_buffer)], dtype=torch.float32).to(device = device)
    actions = torch.tensor([list(action_buffer)], dtype=torch.float32).to(device = device)
    rewards = torch.tensor([100], dtype=torch.float32).reshape(1,1).to(device = device)
    timesteps = torch.tensor([list(timestep_buffer)], dtype=torch.long).to(device = device)

    pred_action, _, _ = model(states, actions, timesteps, rewards, batch_inds = torch.tensor([0]).to(device)) #batch_inds can be any tesnor actually as it won't be used in eval
    action = pred_action.squeeze().detach().cpu().numpy()  # last token prediction

    next_obs, reward, done, truncated, info = env.step(action)
    next_obs = torch.cat([torch.tensor(next_obs["observation"]), 
                          torch.tensor(next_obs["achieved_goal"]), 
                          torch.tensor(next_obs["desired_goal"])],dim = -1).to(device)
    env.render()
    
    total_reward += reward
    print(f"Step {t}, reward={reward:.3f}, total={total_reward:.3f}")

    # Update context
    state_buffer.append(next_obs.cpu().numpy())
    action_buffer.append(action)
    timestep_buffer.append(t + context_len)

    if done or truncated:
        obs = env.reset()[0]
        obs = torch.cat([torch.tensor(obs["observation"]), torch.tensor(obs["achieved_goal"]), torch.tensor(obs["desired_goal"])],dim = -1).to(device)
        state_buffer.clear()
        action_buffer.clear()
        timestep_buffer.clear()
        for t in range(context_len-1):
            state_buffer.append(np.zeros(state_dim))
            action_buffer.append(np.zeros(act_dim))
            timestep_buffer.append(t)

        state_buffer.append(obs.cpu().numpy())
        action_buffer.append(np.zeros(act_dim))
        timestep_buffer.append(context_len)