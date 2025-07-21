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
wandb.init(project="vaelpt-maze2d-training", name="lpt-umaze-run", config={
    "max_len": 100,
    "hidden_size": 128,
    "n_layer": 4,
    "n_head": 1,
    "n_latent": 16,
    "batch_size":64,
    "num_epochs": 2,
    "learning_rate": 1e-4,
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
dataset = minari.load_dataset('D4RL/pointmaze/medium-dense-v2', download=True)



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
        #continue;
        pad_len = context_len - T0
        observations = F.pad(observations, (0, 0, pad_len, 0))
        actions      = F.pad(actions,      (0, 0, pad_len, 0))
        rew          = F.pad(rew.unsqueeze(-1), (0, 0, pad_len, 0)).squeeze(-1)
        done         = F.pad(done,         (pad_len, 0))
        rtg          = F.pad(rtg,          (0, 0, pad_len, 0))
        prev_act     = F.pad(prev_act,     (0, 0, pad_len, 0))
        timesteps    = F.pad(timesteps,    (0, 0, pad_len, 0))

        # 只保存一个片段（整个序列）
        sequence_data.append({
            "observations":  observations,
            "actions":       actions,
            "reward":        rew.unsqueeze(-1),
            "done":          done.unsqueeze(-1),
            "return_to_go":  rtg,
            "prev_actions":  prev_act,
            "timesteps":     timesteps,
        })
    else:
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
    context_len=MAX_LEN,
    n_heads=4,
    drop_p=0.1,
    z_dim =HIDDEN_SIZE*4,
    num_latent=1).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for step in pbar:
        idxs = np.random.choice(len(sequence_data), BATCH_SIZE, replace=False)
        batch = {k: torch.stack([sequence_data[i][k] for i in idxs], dim=0).to(device)
                 for k in sequence_data[0]}
        optimizer.zero_grad()
        (pred_actions,pred_states,pred_rewards),(z_post, mu_post, logvar_post),(z_prior,mu_prior,logvar_prior) = vae.forward(actions=batch["prev_actions"].to(device),
                    states=batch["observations"].to(device),
                    returns_to_go=batch["return_to_go"].to(device),
                    timesteps=batch["timesteps"].to(device),
                    rewards=torch.sum(batch["reward"], dim=1).to(device),
                    disable_test=False)
        # generate_action,generate_state = vae.generate(actions=batch["prev_actions"].to(device),
        #             states=batch["observations"].to(device),
        #             timesteps=batch["timesteps"].to(device),
        #             rewards=torch.sum(batch["reward"], dim=1).to(device))
        
        action_loss = F.mse_loss(pred_actions, batch["actions"].to(device)) 
        reward_loss = F.mse_loss(pred_rewards, torch.sum(batch["reward"], dim=1).to(device))
        recon_loss = action_loss + reward_loss 
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # shape [B, N]
        kl_loss = 0.5 * torch.sum(
        logvar_prior - logvar_post + 
        (torch.exp(logvar_post) + (mu_post - mu_prior)**2) / torch.exp(logvar_prior) - 1,
        dim=-1
        ).mean()
        kl_loss = kl_loss.mean()
        total_loss = recon_loss + 0.01 * epoch * kl_loss
        total_loss.backward()
        with torch.no_grad():
            (pred_actions,pred_states,pred_rewards),(dt_pred_actions,dt_pred_states,dt_pred_rtg),(z,mu,logvar) = vae.forward(actions=batch["prev_actions"].to(device),
            states=batch["observations"].to(device),
            returns_to_go=batch["return_to_go"].to(device),
            timesteps=batch["timesteps"].to(device),
            rewards=torch.sum(batch["reward"], dim=1).to(device),
            disable_test=True)
            disable_action_loss = F.mse_loss(pred_actions, batch["actions"].to(device)) 
        optimizer.step()
        wandb.log({
            "total_loss": total_loss.item(),
            "reward_loss": reward_loss.item(),
            "action_loss": action_loss.item(),
            "random_latent_action_loss": disable_action_loss.item(),
            "kl_loss": kl_loss.item(),
            "epoch": epoch + 1,
            "step": epoch * (len(sequence_data) // BATCH_SIZE) + step,
        })



torch.save(vae,"results/weights/vae_lpt_maze2d.pt")
print("LPT training complete.")


        
        
        
        
        
        
       
