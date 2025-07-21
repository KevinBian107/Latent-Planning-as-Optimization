import torch
import minari
from collections import deque
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

model = torch.load("results/weights/vae_lpt_maze2d.pt", map_location="mps" ,weights_only=False)  # or to('cuda')
device ='mps'

model.to(device)
model.device = device
model.eval()

dataset = minari.load_dataset('D4RL/pointmaze/medium-dense-v2')
#dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download=True)

env = dataset.recover_environment(render_mode="human")
obs = env.reset()[0]
obs = torch.cat([torch.tensor(obs["observation"],dtype=torch.float32), torch.tensor(obs["achieved_goal"],dtype=torch.float32), torch.tensor(obs["desired_goal"],dtype=torch.float32)],dim = -1).to(device)

context_len = 100
state_dim = obs.shape[0]
act_dim = 2

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

STEP = 15000
total_reward = 0
for t in range(STEP):
    states = torch.tensor([list(state_buffer)], dtype=torch.float32).to(device = device)
    actions = torch.tensor([list(action_buffer)], dtype=torch.float32).to(device = device)
    rewards = torch.tensor([50], dtype=torch.float32).reshape(1,1).to(device = device)
    timesteps = torch.tensor([list(timestep_buffer)], dtype=torch.long).to(device = device)
    pred_action,_= model.generate(actions=actions,
            states=states,
            timesteps=timesteps.to(device),
            rewards=rewards,)
    action = pred_action[:,-1,:].squeeze().detach().cpu().numpy()  # last token prediction

    next_obs, reward, done, truncated, info = env.step(action)
    next_obs = torch.cat([torch.tensor(next_obs["observation"],dtype=torch.float32), 
                          torch.tensor(next_obs["achieved_goal"],dtype=torch.float32), 
                          torch.tensor(next_obs["desired_goal"],dtype=torch.float32)],dim = -1).to(device)
    env.render()
    
    total_reward += reward
    print(f"Step {t}, reward={reward:.3f}, total={total_reward:.3f}")

    # Update context
    state_buffer.append(next_obs.cpu().numpy())
    action_buffer.append(action)
    timestep_buffer.append(t + context_len)

    if done or truncated:
        obs = env.reset()[0]
        obs = torch.cat([torch.tensor(obs["observation"],dtype=torch.float32), torch.tensor(obs["achieved_goal"],dtype=torch.float32), torch.tensor(obs["desired_goal"],dtype=torch.float32)],dim = -1).to(device)
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
