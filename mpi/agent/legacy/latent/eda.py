import wandb
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari
import torch.nn.functional as F

import torch.nn as nn
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from agent.src.layers.block import Block

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
###
device = torch.device("mps")
context_len = 250
h_dim = 128
state_dim = 8
act_dim = 2
n_blocks = 4
drop_p = 0.1
n_heads = 1
BATCH_SIZE = 64
EPOCH = 2
LEARNING_RATE = 1e-5
###
###
class DTdecoder(nn.Module):
    def __init__(self,h_dim,state_dim,act_dim,use_action_tanh = True):
        super().__init__()
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else [])))
    def forward(self,h):

        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1]) 
        return action_preds,state_preds,return_preds
###
class DecisionTransformerEncoder(nn.Module):

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.apply(self._init_weights)


    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
        h = self.transformer(h)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        h_seq = h.reshape(B, 3 * T, self.h_dim)  # [B, L, H]

        return h_seq,h
#
datasets = ["D4RL/pointmaze/umaze-dense-v2",'D4RL/pointmaze/open-dense-v2','D4RL/pointmaze/medium-dense-v2','D4RL/pointmaze/large-dense-v2']
all_trajectories = []

# Load each dataset and add source labels
for dataset_name in datasets:
    print(dataset_name)
    dataset = minari.load_dataset(dataset_name, download=True)
    for episode in tqdm(dataset, desc=f"Loading {dataset_name} episodes"):
        obs_dict = episode.observations
        obs = obs_dict['observation'][:-1] 
        desired_goal = obs_dict['desired_goal'][:-1] 
        achieved_goal = obs_dict['achieved_goal'][:-1]
        
        full_state_space = np.concatenate([obs, desired_goal, achieved_goal], axis=1)
        T0 = full_state_space.shape[0]

        observations = torch.tensor(full_state_space, dtype=torch.float32, device=device)
        actions = torch.tensor(episode.actions[:T0], dtype=torch.float32, device=device)
        rew = torch.tensor(episode.rewards[:T0], dtype=torch.float32, device=device)
        done = torch.tensor(episode.terminations[:T0], dtype=torch.bool, device=device)

        rtg = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
        prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
        timesteps = torch.arange(T0, dtype=torch.long, device=device).unsqueeze(-1)

        if T0 < context_len:
            pad_len = context_len - T0
            observations = F.pad(observations, (0, 0, pad_len, 0))
            actions = F.pad(actions, (0, 0, pad_len, 0))
            rew = F.pad(rew.unsqueeze(-1), (0, 0, pad_len, 0)).squeeze(-1)
            done = F.pad(done, (pad_len, 0))
            rtg = F.pad(rtg, (0, 0, pad_len, 0))
            prev_act = F.pad(prev_act, (0, 0, pad_len, 0))
            timesteps = F.pad(timesteps, (0, 0, pad_len, 0))

            # Save a single sequence
            all_trajectories.append({
                "observations": observations,
                "actions": actions,
                "reward": rew.unsqueeze(-1),
                "done": done.unsqueeze(-1),
                "return_to_go": rtg,
                "prev_actions": prev_act,
                "timesteps": timesteps,
            })
        else:
            for i in range(T0 - context_len + 1):
                all_trajectories.append({
                    "observations": observations[i:i+context_len],
                    "actions": actions[i:i+context_len],
                    "reward": rew[i:i+context_len].unsqueeze(-1),
                    "done": done[i:i+context_len].unsqueeze(-1),
                    "return_to_go": rtg[i:i+context_len],
                    "prev_actions": prev_act[i:i+context_len],
                    "timesteps": timesteps[i:i+context_len],
                })

print(f"Loaded {len(all_trajectories)} sequences across all datasets.")
random.shuffle(all_trajectories)
# Stratified sampling based on total rewards
wandb.init(project="dt_based_mixed_training", name="dt-mixed-run", config={
    "max_len": context_len,
    "hidden_size": h_dim,
    "n_layer": n_blocks,
    "n_head": n_heads,
    "batch_size":BATCH_SIZE,
    "num_epochs": EPOCH,
    "learning_rate": LEARNING_RATE,
})
sequence_data = all_trajectories
encoder = DecisionTransformerEncoder(h_dim=h_dim,state_dim=state_dim,act_dim=act_dim,n_blocks=n_blocks,context_len=context_len,n_heads=n_heads,drop_p=drop_p).to(device)
decoder = DTdecoder(h_dim=h_dim,state_dim=state_dim,act_dim=act_dim).to(device)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCH):
    pbar = tqdm(range(len(sequence_data) // BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for step in pbar:
        idxs = np.random.choice(len(sequence_data), BATCH_SIZE, replace=False)
        batch = {k: torch.stack([sequence_data[i][k] for i in idxs], dim=0).to(device)
                for k in sequence_data[0]}
        h_sequence,h = encoder.forward(timesteps=batch["timesteps"].squeeze(-1).to(device=device), 
                        states=batch["observations"].to(device=device),
                        actions=batch["prev_actions"].to(device = device),
                        returns_to_go=batch["return_to_go"].to(device = device))
        action_preds,state_preds,return_preds = decoder.forward(h)
        mse = F.mse_loss(batch["actions"].to(device = device),action_preds)
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        mse.backward()
        encoder_optim.step()
        decoder_optim.step()
        wandb.log({
        "action_loss": mse.item(),
        "epoch": epoch + 1,
        "step": epoch * (len(sequence_data) // BATCH_SIZE) + step,
        })

torch.save(encoder,"results/weights/dt_mixed_maze2d.pt")
print("DT training complete.")

            
            




    
