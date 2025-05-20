import torch
import minari
from collections import deque
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- 环境和模型设置 ---
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Load model
model = torch.load("results/weights/dt_mujoco.pt",weights_only=False)
model.to(device)
model.device = device
model.eval()

# Load env
dataset = minari.load_dataset('mujoco/halfcheetah/expert-v0')
env = dataset.recover_environment(render_mode="human")
obs = env.reset()[0]

# --- 推理缓存设置 ---
max_len = 50  # 最多考虑多少步的历史
state_dim = obs.shape[0]
action_dim = env.action_space.shape[0]
return_to_go = 12000.0  # 可以是一个经验值

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

for _ in range(10000):
    # 更新缓存
    states.append(torch.tensor(obs, dtype=torch.float32))
    actions.append(torch.zeros(action_dim))  # dummy action for initial inference
    rewards.append(0.0)  # dummy reward
    timesteps.append(t)
    rtgs.append(torch.tensor([return_to_go - total_reward], dtype=torch.float32))

    # 构造 batch（pad 到 max_len）
    def pad(seq, size, dim):
        pad_len = size - len(seq)
        if pad_len <= 0:
            return torch.stack(list(seq)[-size:])
        else:
            stacked = torch.stack(list(seq))  # ensure it's 2D
            if stacked.ndim == 1:
                stacked = stacked.unsqueeze(-1)  # make it [N, 1]
            return torch.cat([
                torch.zeros((pad_len, dim)),
                stacked
            ], dim=0)

    states_tensor = pad(states, max_len, state_dim).unsqueeze(0).to(device)
    actions_tensor = pad(actions, max_len, action_dim).unsqueeze(0).to(device)
    rtgs_tensor = pad(rtgs, max_len, 1).unsqueeze(0).to(device)
    #timesteps_tensor = torch.tensor([timesteps], dtype=torch.long).to(device)
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



    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    actions[-1] = torch.tensor(action)  # 更新真实动作
    rewards[-1] = reward

    t += 1
    if done:
        break

