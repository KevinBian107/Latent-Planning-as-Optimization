import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")


from src.models.LPT import LatentPlannerModel

all_losses = []
r_losses = []
a_losses = []

MAX_LEN = 50
HIDDEN_SIZE = 24
N_LAYER = 3
N_HEAD = 1
BATCH_SIZE = 128
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4

context_len = MAX_LEN

dataset = minari.load_dataset('mujoco/halfcheetah/expert-v0', download=True)
env = dataset.recover_environment()

sequence_data = []

for episode in tqdm(dataset):
    observations = torch.tensor(episode.observations[:-1], dtype=torch.float32).to(device)
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

model = LatentPlannerModel(
    state_dim=observations.shape[1],
    act_dim=actions.shape[1],
    h_dim=HIDDEN_SIZE,
    context_len=context_len,
    n_blocks=N_LAYER,
    n_heads=N_HEAD,
    n_latent=6,
    device=device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE))
    for _ in pbar:
        batch_list = np.random.choice(sequence_data, BATCH_SIZE, replace=False)
        batch = {k: torch.stack([item[k] for item in batch_list], dim=0).to(device) for k in batch_list[0]}
        batch_inds = torch.arange(BATCH_SIZE, device=device)

        # Forward with latent z
        pred_action, pred_state, pred_reward = model(
            states=batch["observations"],
            actions=batch["prev_actions"],
            timesteps=batch["timesteps"].squeeze(-1),
            rewards=torch.sum(batch["reward"],dim=1),
            batch_inds=batch_inds,
        )

        optimizer.zero_grad()
        # 可以只用 reward loss，也可以 action + reward 一起用
        loss_r = torch.nn.MSELoss()(pred_reward, torch.sum(batch["reward"],dim = 1).squeeze(1))
        loss_a = torch.nn.MSELoss()(pred_action, batch["actions"][:, -1])
        loss = 0.25 * loss_r + loss_a
        loss.backward()
        optimizer.step()
        all_losses.append(loss.item())
        r_losses.append(loss_r.item())
        a_losses.append(loss_a.item())

        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss_a.item():.4f}")
torch.save(model,"results/weights/lpt_mujoco.pt")
print("LPT training complete.")
plt.figure(figsize=(8, 4))
plt.plot(all_losses, label="Total loss")
plt.plot(a_losses, label="Action loss")
plt.plot(r_losses, label="Reward loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("LPT Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
