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
import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
from agent.src.models.lpt import LatentPlannerModel
from agent.src.models.lpt_experiment import VAE
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")
# Initialize wandb
wandb.init(project="lpt-maze2d-training", name="lpt-umaze-run", config={
    "max_len": 150,
    "hidden_size": 512,
    "n_layer": 4,
    "n_head": 1,
    "n_latent": 16,
    "batch_size":32,
    "num_epochs": 4,
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

vae = VAE(state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=4,
    h_dim=HIDDEN_SIZE,
    context_len=150,
    n_heads=1,
    drop_p=0.1,
    z_dim = HIDDEN_SIZE,
    num_latent=1).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for step in pbar:
        idxs = np.random.choice(len(sequence_data), BATCH_SIZE, replace=False)
        batch = {k: torch.stack([sequence_data[i][k] for i in idxs], dim=0).to(device)
                 for k in sequence_data[0]}
        optimizer.zero_grad()
        (pred_actions,pred_states,pred_rewards),(z,mu,logvar) = vae.forward(actions=batch["prev_actions"].to(device),
                    states=batch["observations"].to(device),
                    returns_to_go=batch["return_to_go"].to(device),
                    timesteps=batch["timesteps"].to(device),
                    rewards=None,disable_test=True)
        
        action_loss = F.mse_loss(pred_actions, batch["actions"].to(device)) 
        reward_loss = F.mse_loss(pred_rewards, batch["reward"].to(device))
        recon_loss = action_loss + reward_loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # shape [B, N]
        kl_loss = kl_loss.mean()

        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()
        wandb.log({
            "total_loss": total_loss.item(),
            "reward_loss": reward_loss.item(),
            "action_loss": action_loss.item(),
            "kl_loss": kl_loss.item(),
            "epoch": epoch + 1,
            "step": epoch * (len(sequence_data) // BATCH_SIZE) + step,
        })


        
        
        
        
        
        
       
