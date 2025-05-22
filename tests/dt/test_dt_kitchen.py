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
# -------------------- 超参数 --------------------
MAX_LEN = 150
HIDDEN_SIZE = 32
N_LAYER = 4
N_HEAD = 1
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
    obs = episode.observations['observation'][:-1]
    desired_goal = episode.observations['desired_goal']
    achieved_goal = episode.observations['achieved_goal']
    
    task_keys = ['microwave', 'kettle', 'light switch', 'bottom burner']

    desired_goals_list = [desired_goal[key][:-1] for key in task_keys]
    achieved_goals_list = [achieved_goal[key][:-1] for key in task_keys]

    all_desired_goals = np.concatenate(desired_goals_list, axis=1)  # shape: (seq_len, sum(goal_dims))
    all_achieved_goals = np.concatenate(achieved_goals_list, axis=1)  # shape: (seq_len, sum(goal_dims))
    
    full_state_space = np.concatenate([obs, all_desired_goals, all_achieved_goals], axis=1)
    
    # full state space shape = [timestep, 12 + 12 + 59]

    observations = torch.tensor(full_state_space, dtype=torch.float32).to(device)
    actions = torch.tensor(episode.actions, dtype=torch.float32).to(device)
    rew = torch.tensor(episode.rewards, dtype=torch.float32).to(device)
    done = torch.tensor(episode.terminations, dtype=torch.bool).to(device)

    rtg = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
    prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
    timesteps = torch.arange(len(observations), dtype=torch.long, device=device).unsqueeze(-1)

    if observations.shape[0] < context_len:
        continue

    for i in range(observations.shape[0] - context_len + 1):
        sequence_data.append({
            "observations": observations[i:i+context_len],
            "actions": actions[i:i+context_len],
            "reward": rew[i:i+context_len].unsqueeze(-1),
            "done": done[i:i+context_len].unsqueeze(-1),
            "return_to_go": rtg[i:i+context_len],
            "prev_actions": prev_act[i:i+context_len],
            "timesteps": timesteps[i:i+context_len],
        })
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