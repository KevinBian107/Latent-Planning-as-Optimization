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
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

# -------------------- 超参数 --------------------
MAX_LEN = 5
HIDDEN_SIZE = 32
N_LAYER = 3
N_HEAD = 1
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4

context_len = MAX_LEN

# -------------------- 加载数据 --------------------
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2', download=True)
env = dataset.recover_environment()

# 改为 List 缓存训练段
sequence_data = []

for episode in tqdm(dataset):
    observations = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32).to(device)
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