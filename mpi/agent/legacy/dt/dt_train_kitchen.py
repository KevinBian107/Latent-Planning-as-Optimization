import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari

all_losses = []
r_losses = []
a_losses = []
# -------------------- 设置设备 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

# -------------------- 工作路径 --------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.process_obs import process_observation, kitchen_goal_obs_dict
# -------------------- 超参数 --------------------
MAX_LEN = 50
HIDDEN_SIZE = 256
N_LAYER = 4
N_HEAD = 8
BATCH_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

context_len = MAX_LEN

# -------------------- 加载数据 --------------------
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2', download=True)
# env = dataset.recover_environment()

# 改为 List 缓存训练段
sequence_data = []

for episode in tqdm(dataset):
    obs = episode.observations
    observation, desired_goal, achieved_goal = process_observation(kitchen_goal_obs_dict, obs).values()
    full_state_space = torch.cat([observation, desired_goal, achieved_goal], dim=-1)
    observations = torch.tensor(full_state_space, dtype=torch.float32).to(device)
    actions = torch.tensor(episode.actions, dtype=torch.float32).to(device)
    rew = torch.tensor(episode.rewards, dtype=torch.float32).to(device)
    done = torch.tensor(episode.terminations, dtype=torch.bool).to(device)
    if len(observations) < 40:
        continue;
    rtg = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
    prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
    timesteps = torch.arange(len(observations), dtype=torch.long, device=device).unsqueeze(-1)

    T = observations.shape[0]
    
    if T < context_len:
        # 左侧 zero padding 到 context_len
        pad_len = context_len - T
        pad = lambda x, dim: torch.cat([torch.zeros(pad_len, *x.shape[1:], device=x.device), x], dim=dim)

        seg = {
            "observations": pad(observations, 0),
            "actions": pad(actions, 0),
            "reward": pad(rew.unsqueeze(-1), 0),
            "done": pad(done.unsqueeze(-1).float(), 0),  # later convert back to bool if needed
            "return_to_go": pad(rtg, 0),
            "prev_actions": pad(prev_act, 0),
            "timesteps": pad(timesteps, 0),
        }
        sequence_data.append(seg)
        for k, v in seg.items():
            if v.shape[0] != context_len:
                print(f"[DEBUG] T = {T}")
                for _k, _v in seg.items():
                    print(f" - {repr(_k)} shape = {_v.shape}")
                raise ValueError(f"[INVALID SEGMENT] {k} has shape {v.shape}, expected {context_len}")
    else:
        # 正常滑窗处理
        T = min(
                observation.shape[0],
                actions.shape[0],
                rew.shape[0],
                done.shape[0],
                desired_goal.shape[0],
                achieved_goal.shape[0]
            )
        observation = observation[:T]
        desired_goal = desired_goal[:T]
        achieved_goal = achieved_goal[:T]
        actions = actions[:T]
        rew = rew[:T]
        done = done[:T]
        for i in range(T - context_len + 1):
            seg = {
                "observations": observations[i:i+context_len],
                "actions": actions[i:i+context_len],
                "reward": rew[i:i+context_len].unsqueeze(-1),
                "done": done[i:i+context_len].unsqueeze(-1).float(),
                "return_to_go": rtg[i:i+context_len],
                "prev_actions": prev_act[i:i+context_len],
                "timesteps": timesteps[i:i+context_len],
            }
            sequence_data.append(seg)
            for k, v in seg.items():
                if v.shape[0] != context_len:
                    print(f"[DEBUG] T = {T}")
                    for _k, _v in seg.items():
                        print(f" - {repr(_k)} shape = {_v.shape}")
                    raise ValueError(f"[INVALID SEGMENT] {k} has shape {v.shape}, expected {context_len}")

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    # axs[0].plot(timesteps.cpu(), rtg.cpu(), label='Return-to-Go', color='blue')
    # axs[0].set_title("RTG Decay Over Time in One Episode")
    # axs[0].set_xlabel("Timestep")
    # axs[0].set_ylabel("RTG Value")
    # axs[0].grid(True)
    # axs[0].legend()

    # # 下图：RTG 值分布
    # axs[1].hist(rtg.cpu(),density=True)
    # axs[1].set_title("RTG Value Distribution")
    # axs[1].set_xlabel("RTG Value")
    # axs[1].set_ylabel("Frequency")
    # axs[1].grid(True)

    # plt.tight_layout()
    # plt.show()

print(f"Loaded {len(sequence_data)} sequences.")

# -------------------- 初始化模型 --------------------
from src.models.decision_transformer import DecisionTransformer

model = DecisionTransformer(
    state_dim=observations.shape[1],
    act_dim=actions.shape[1],
    n_blocks=N_LAYER,
    h_dim=HIDDEN_SIZE,
    context_len=context_len,
    n_heads=N_HEAD,
    drop_p=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 训练循环 --------------------


for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE))
    for _ in pbar:
        # 手动采样 BATCH_SIZE 条 segment
        batch_list = np.random.choice(sequence_data, BATCH_SIZE, replace=False)
        batch = {k: torch.stack([item[k] for item in batch_list], dim=0).to(device) for k in batch_list[0]}

        state_preds, action_preds, return_preds = model(
            timesteps=batch["timesteps"].squeeze(-1),
            states=batch["observations"],
            actions=batch["prev_actions"],
            returns_to_go=batch["return_to_go"]
        )

        optimizer.zero_grad()
        loss = torch.nn.MSELoss()(action_preds, batch["actions"])
        
        loss.backward()
        all_losses.append(loss.item())
        optimizer.step()

        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
torch.save(model,"results/weights/dt_kitchen.pt")
print("Training complete.")
plt.figure(figsize=(8, 4))
plt.plot(all_losses, label="Action loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("LPT Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()