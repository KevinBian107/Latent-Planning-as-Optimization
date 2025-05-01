import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import minari

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.MPILL import MPILearningLearner
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 15
HIDDEN_SIZE = 32
BATCH_SIZE = 16
N_EPOCHS = 5
ema_momentum = 0.99
alpha_bar = None

# Load Dataset
dataset = minari.load_dataset("D4RL/kitchen/mixed-v2", download=True)
env = dataset.recover_environment()
sequence_data = []

for episode in tqdm(dataset):
    obs = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32)
    acts = torch.tensor(episode.actions, dtype=torch.float32)
    rews = torch.tensor(episode.rewards, dtype=torch.float32)
    rtg = rews.flip([0]).cumsum(0).flip([0]).unsqueeze(-1)
    prev_acts = torch.cat([torch.zeros_like(acts[:1]), acts[:-1]], dim=0)
    timesteps = torch.arange(len(obs)).unsqueeze(-1)
    if obs.shape[0] < MAX_LEN: continue
    for i in range(obs.shape[0] - MAX_LEN + 1):
        sequence_data.append({
            "observations": obs[i:i+MAX_LEN],
            "actions": acts[i:i+MAX_LEN],
            "reward": rews[i:i+MAX_LEN].unsqueeze(-1),
            "return_to_go": rtg[i:i+MAX_LEN],
            "prev_actions": prev_acts[i:i+MAX_LEN],
            "timesteps": timesteps[i:i+MAX_LEN],
        })

print(f"Loaded {len(sequence_data)} sequences.")
model = MPILearningLearner(state_dim=obs.shape[1], act_dim=acts.shape[1], context_len=MAX_LEN, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses, alpha_losses, kl_losses = [], [], []

for epoch in range(N_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE))
    for _ in pbar:
        batch = np.random.choice(sequence_data, BATCH_SIZE, replace=False)
        batch = {k: torch.stack([d[k] for d in batch]).to(device) for k in batch[0]}

        optimizer.zero_grad()
        T = batch["observations"].shape[1]
        timesteps = batch["timesteps"][:, :T].squeeze(-1)

        pred_action, pred_reward, alpha_k, alpha_loss, kl = model(
            batch["observations"], batch["prev_actions"], batch["reward"], timesteps, alpha_bar=alpha_bar
        )

        loss_r = F.mse_loss(pred_reward, batch["reward"][:, -1, 0])
        loss_a = F.mse_loss(pred_action, batch["actions"][:, -1])
        total_loss = loss_r + loss_a + 0.01 * kl + 0.01 * alpha_loss

        total_loss.backward()
        optimizer.step()

        # Update alpha_bar
        with torch.no_grad():
            alpha_k_mean = alpha_k.mean(dim=0)
            alpha_bar = alpha_k_mean if alpha_bar is None else ema_momentum * alpha_bar + (1 - ema_momentum) * alpha_k_mean

        losses.append(total_loss.item())
        alpha_losses.append(alpha_loss.item())
        kl_losses.append(kl.item())
        pbar.set_description(f"Epoch {epoch+1} Loss={total_loss.item():.3f}")

# Plot
plt.plot(losses, label="Total Loss")
plt.plot(alpha_losses, label="Alpha Supervision")
plt.plot(kl_losses, label="KL(zeta)")
plt.title("TD-BU MPI Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
