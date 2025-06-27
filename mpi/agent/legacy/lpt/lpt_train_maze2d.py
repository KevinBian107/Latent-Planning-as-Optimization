import wandb
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.contrastive_loss import reward_latent_consistency_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agent.src.models.lpt import LatentPlannerModel
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")
# Initialize wandb
wandb.init(project="lpt-maze2d-training", name="lpt-umaze-run", config={
    "max_len": 150,
    "hidden_size": 32,
    "n_layer": 4,
    "n_head": 1,
    "n_latent": 16,
    "batch_size":16,
    "num_epochs": 2,
    "learning_rate": 3e-5,
    "z_n_iters":3,
})

MAX_LEN = wandb.config.max_len
HIDDEN_SIZE = wandb.config.hidden_size
N_LAYER = wandb.config.n_layer
N_HEAD = wandb.config.n_head
N_LATENT = wandb.config.n_latent
BATCH_SIZE = wandb.config.batch_size
NUM_EPOCHS = wandb.config.num_epochs
LEARNING_RATE = wandb.config.learning_rate
Z_N_ITERS = wandb.config.z_n_iters
context_len = MAX_LEN
dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download=True)


sequence_data = []
for episode in tqdm(dataset, desc="Loading U-Maze episodes"):
    obs_dict = episode.observations
    obs            = obs_dict['observation'][:-1] 
    desired_goal   = obs_dict['desired_goal'][:-1] 
    achieved_goal  = obs_dict['achieved_goal'][:-1]

    full_state_space = np.concatenate([obs, desired_goal, achieved_goal], axis=1)
    T0 = full_state_space.shape[0]

    observations = torch.tensor(full_state_space, dtype=torch.float32, device=device)
    actions      = torch.tensor(episode.actions[:T0],    dtype=torch.float32, device=device)
    rew          = torch.tensor(episode.rewards[:T0],    dtype=torch.float32, device=device)
    done         = torch.tensor(episode.terminations[:T0],dtype=torch.bool,    device=device)

    rtg         = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
    prev_act    = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
    timesteps   = torch.arange(T0, dtype=torch.long, device=device).unsqueeze(-1)

    if T0 < context_len:
        continue

    for i in range(T0 - context_len + 1):
        sequence_data.append({
            "observations":  observations[i:i+context_len],
            "actions":       actions[i:i+context_len],
            "reward":        rew[i:i+context_len].unsqueeze(-1),
            "done":          done[i:i+context_len].unsqueeze(-1),
            "return_to_go":  rtg[i:i+context_len],
            "prev_actions":  prev_act[i:i+context_len],
            "timesteps":     timesteps[i:i+context_len],
        })

print(f"Loaded {len(sequence_data)} sequences from maze2d-umaze.")

state_dim = sequence_data[0]["observations"].shape[-1]
act_dim   = sequence_data[0]["actions"].shape[-1]

model = LatentPlannerModel(
    state_dim=state_dim,
    act_dim=act_dim,
    h_dim=HIDDEN_SIZE,
    context_len=context_len,
    n_blocks=N_LAYER,
    n_heads=N_HEAD,
    n_latent=N_LATENT,
    z_n_iters=Z_N_ITERS,
    device=device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

all_losses = []
r_losses   = []
a_losses   = []
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for step in pbar:
        idxs = np.random.choice(len(sequence_data), BATCH_SIZE, replace=False)
        batch = {
            k: torch.stack([sequence_data[i][k] for i in idxs], dim=0).to(device)
            for k in sequence_data[0]
        }
        batch_inds = torch.arange(BATCH_SIZE, device=device)

        pred_action, pred_state, pred_reward, z_latent = model(
            states=batch["observations"],
            actions=batch["prev_actions"],
            timesteps=batch["timesteps"].squeeze(-1),
            rewards=torch.sum(batch["reward"], dim=1),
            batch_inds=batch_inds,
        )

        loss_r = torch.nn.MSELoss()(pred_reward, torch.sum(batch["reward"], dim=1).squeeze(1))
        loss_a = torch.nn.MSELoss()(pred_action,  batch["actions"][:, -1])
        reward_contrastive_loss  = reward_latent_consistency_loss(z_latent,pred_reward)
        loss   =  3 * loss_r + loss_a + reward_contrastive_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        r_losses.append(loss_r.item())
        a_losses.append(loss_a.item())

        pbar.set_postfix(loss=f"{loss.item():.4f}", pred_rewards=f"{pred_reward.mean().item():.4f}")

        # Compute batch statistics
        pred_reward_values = pred_reward.detach().cpu()
        pred_reward_mean = pred_reward_values.mean().item()
        pred_reward_var  = pred_reward_values.var().item()

        actual_reward_values = torch.sum(batch["reward"], dim=1).squeeze(1).detach().cpu()
        actual_reward_mean = actual_reward_values.mean().item()
        actual_reward_var  = actual_reward_values.var().item()
        
        # Log to wandb
        wandb.log({
            "total_loss": loss.item(),
            "reward_loss": loss_r.item(),
            "action_loss": loss_a.item(),
            "predicted_reward_mean": pred_reward_mean,
            "predicted_reward_var": pred_reward_var,
            "actual_reward_mean": actual_reward_mean,
            "actual_reward_var": actual_reward_var,
            "epoch": epoch + 1,
            "step": epoch * (len(sequence_data) // BATCH_SIZE) + step,
            "reward_contrastive_loss": reward_contrastive_loss,
        })

# Save model
os.makedirs("results/weights", exist_ok=True)
torch.save(model, "results/weights/lpt_maze2d_umaze.pt")
print("LPT training complete. Weights saved to results/weights/lpt_maze2d_umaze.pt")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(all_losses, label="Total loss")
plt.plot(a_losses, label="Action loss")
plt.plot(r_losses, label="Reward loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("LPT Training on Maze2D U-Maze")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Finish wandb run
wandb.finish()
