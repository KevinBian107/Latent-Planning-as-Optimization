import torch
import minari
from collections import deque
import numpy as np
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.process_obs import process_observation,kitchen_goal_obs_dict
# --- 环境和模型设置 ---
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Load model
model = torch.load("results/weights/dt_kitchen.pt",weights_only=False)
model.to(device)
model.device = device
model.eval()

# Load env
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
env = dataset.recover_environment(render_mode="human",eval_env=True)
obs = env.reset()[0]
observation,desired_goal,achieved_goal = process_observation(kitchen_goal_obs_dict,obs).values()
full_state_obs = torch.cat([observation, desired_goal, achieved_goal])
# --- 推理缓存设置 ---
max_len = 50  # 最多考虑多少步的历史
state_dim = full_state_obs.shape[0]
action_dim = env.action_space.shape[0]
return_to_go = 400.0  # 

states = deque([], maxlen=max_len)
actions = deque([], maxlen=max_len)
rewards = deque([], maxlen=max_len)
timesteps = deque([], maxlen=max_len)
rtgs = deque([], maxlen=max_len)

total_reward = 0
t = 0

def pad_timestep(seq, size):
    pad_len = size - len(seq)
    if pad_len <= 0:
        return torch.tensor(list(seq)[-size:], dtype=torch.long)
    else:
        return torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.tensor(list(seq), dtype=torch.long)
        ])
    

for i in range(max_len):
    states.append(torch.tensor(full_state_obs, dtype=torch.float32))
    actions.append(torch.zeros(action_dim))  # dummy action for initial inference
    rewards.append(0.0)  # dummy reward
    timesteps.append(t)
    rtgs.append(torch.tensor([return_to_go - total_reward], dtype=torch.float32))


#timesteps_tensor = torch.tensor([timesteps], dtype=torch.long).to(device)
for _ in range(5000):
    def pad(seq, size, dim):
        pad_len = size - len(seq)
        if pad_len <= 0:
            return torch.stack(list(seq)[-size:])
        else:
            stacked = torch.stack(list(seq))
            return torch.cat([
                torch.zeros((pad_len, dim), dtype=torch.float32),
                stacked
            ], dim=0)
    states_tensor = pad(states, max_len, state_dim).unsqueeze(0).to(device)
    actions_tensor = pad(actions, max_len, action_dim).unsqueeze(0).to(device)
    rtgs_tensor = pad(rtgs, max_len, 1).unsqueeze(0).to(device)
    timesteps_tensor = pad_timestep(timesteps, max_len).unsqueeze(0).to(device)

    # --- 模型前向 ---
    with torch.no_grad():
        outputs = model.forward(
            timesteps=timesteps_tensor,
            states=states_tensor,
            actions=actions_tensor,
            returns_to_go=rtgs_tensor
        )
        action_preds = outputs[1]  # 第 1 个是 action_preds
        action = action_preds[0, -1].cpu().numpy()
    obs_dict, reward, done, _, _ = env.step(action)
    observation,desired_goal,achieved_goal = process_observation(kitchen_goal_obs_dict,obs_dict).values()
    full_state_obs = torch.cat([observation, desired_goal, achieved_goal],dim = -1)
    total_reward += reward
    if reward:
        print(total_reward)

    actions.popleft()
    states.popleft()
    rewards.popleft()
    rtgs.popleft()
    timesteps.popleft()

    actions.append(torch.tensor(action, dtype=torch.float32))
    states.append(torch.tensor(full_state_obs, dtype=torch.float32))
    rewards.append(reward)
    rtgs.append(torch.tensor([rtgs[-1][0] - reward], dtype=torch.float32))
    timesteps.append(t)
    t += 1
    if done:
        break

