import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari

# -------------------- 设置设备 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

# -------------------- 工作路径 --------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# -------------------- 超参数 --------------------
MAX_LEN       = 150
HIDDEN_SIZE   = 256
N_LAYER       = 4
N_HEAD        = 1
BATCH_SIZE    = 128
NUM_EPOCHS    = 2
LEARNING_RATE = 1e-4

context_len = MAX_LEN

# -------------------- 加载 Maze2D U-Maze 数据 --------------------
dataset = minari.load_dataset('D4RL/pointmaze/medium-dense-v2', download=True)

sequence_data = []
for episode in tqdm(dataset, desc="Loading U-Maze episodes"):
    obs_dict = episode.observations
    # Slice off last timestep so everything stays aligned
    obs           = obs_dict['observation'][:-1]   # (T-1, obs_dim)
    desired_goal  = obs_dict['desired_goal'][:-1]  # (T-1, goal_dim)
    achieved_goal = obs_dict['achieved_goal'][:-1] # (T-1, goal_dim)

    # Concatenate into full state vector
    full_state_space = np.concatenate([obs, desired_goal, achieved_goal], axis=1)
    T0 = full_state_space.shape[0]

    # Build tensors, slicing actions/rewards/dones to T0
    observations = torch.tensor(full_state_space, dtype=torch.float32, device=device)
    actions      = torch.tensor(episode.actions[:T0],    dtype=torch.float32, device=device)
    rewards      = torch.tensor(episode.rewards[:T0],    dtype=torch.float32, device=device)
    done         = torch.tensor(episode.terminations[:T0],dtype=torch.bool,    device=device)

    # Compute return-to-go, prev_actions, timesteps
    rtg          = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
    prev_actions = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
    timesteps    = torch.arange(T0, dtype=torch.long, device=device).unsqueeze(-1)

    # Skip episodes shorter than context window
    if T0 < context_len:
        continue

    # Create sliding-window segments
    for i in range(T0 - context_len + 1):
        sequence_data.append({
            "observations":  observations[i:i+context_len],
            "actions":       actions[i:i+context_len],
            "reward":        rewards[i:i+context_len].unsqueeze(-1),
            "done":          done[i:i+context_len].unsqueeze(-1).float(),
            "return_to_go":  rtg[i:i+context_len],
            "prev_actions":  prev_actions[i:i+context_len],
            "timesteps":     timesteps[i:i+context_len],
        })

print(f"Loaded {len(sequence_data)} sequences from maze2d-umaze.")

# -------------------- 初始化模型 --------------------
from src.models.decision_transformer import DecisionTransformer

state_dim = sequence_data[0]["observations"].shape[-1]
act_dim   = sequence_data[0]["actions"].shape[-1]

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=N_LAYER,
    h_dim=HIDDEN_SIZE,
    context_len=context_len,
    n_heads=N_HEAD,
    drop_p=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 训练循环 --------------------
all_losses = []

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for _ in pbar:
        idxs = np.random.choice(len(sequence_data), BATCH_SIZE, replace=False)
        batch = {k: torch.stack([sequence_data[i][k] for i in idxs], dim=0).to(device)
                 for k in sequence_data[0]}

        _, action_preds, _ = model(
            timesteps=batch["timesteps"].squeeze(-1),
            states=batch["observations"],
            actions=batch["prev_actions"],
            returns_to_go=batch["return_to_go"]
        )

        loss = torch.nn.MSELoss()(action_preds, batch["actions"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

# -------------------- 保存模型 & 可视化 --------------------
os.makedirs("results/weights", exist_ok=True)
torch.save(model, "results/weights/dt_maze2d_dense_umaze.pt")
print("Training complete. Weights saved to results/weights/dt_maze2d_umaze.pt")

plt.figure(figsize=(8, 4))
plt.plot(all_losses, label="Action MSE")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Decision Transformer on Maze2D U-Maze")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
