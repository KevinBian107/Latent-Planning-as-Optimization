import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,IncrementalPCA
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import random
import plotly.express as px
import pandas as pd
import minari
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Sampler
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from agent.src.models.lpt_experiment import DecisionTransformerEncoder
# Load the trained encoder model
encoder = torch.load("results/weights/dt_mixed_maze2d.pt",map_location="mps" ,weights_only=False)
encoder.eval()
###
device = torch.device("mps")
context_len = 150
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
# Prepare the datasets again
datasets = ["D4RL/pointmaze/umaze-dense-v2",'D4RL/pointmaze/open-dense-v2','D4RL/pointmaze/medium-dense-v2','D4RL/pointmaze/large-dense-v2']
all_trajectories = []

# Load the dataset and generate h_sequence
class MinariTrajectoryDataset(Dataset):
    def __init__(self, datasets, context_len=150, device='cpu'):
        self.context_len = context_len
        self.device = device
        self.slice_infos = []  # 每个元素是: (dataset_name, episode_index, slice_start_index)

        # 记录所有的 dataset 和 episode 的切片信息
        self.datasets = {name: minari.load_dataset(name, download=True) for name in datasets}
        for dataset_name, ds in self.datasets.items():
            for ep_idx, episode in enumerate(tqdm(ds, desc=f"Indexing {dataset_name}")):
                obs_len = len(episode.observations['observation']) - 1  # 去掉最后一个
                if obs_len < context_len:
                    self.slice_infos.append((dataset_name, ep_idx, -1))  # -1 表示需要 pad
                else:
                    for i in range(obs_len - context_len + 1):
                        self.slice_infos.append((dataset_name, ep_idx, i))

    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, idx):
        dataset_name, ep_idx, start = self.slice_infos[idx]
        episode = self.datasets[dataset_name][ep_idx]

        obs_dict = episode.observations
        obs = obs_dict['observation'][:-1]
        desired_goal = obs_dict['desired_goal'][:-1]
        achieved_goal = obs_dict['achieved_goal'][:-1]

        full_state_space = np.concatenate([obs, desired_goal, achieved_goal], axis=1)
        T0 = full_state_space.shape[0]

        observations = torch.tensor(full_state_space, dtype=torch.float32, device=self.device)
        actions = torch.tensor(episode.actions[:T0], dtype=torch.float32, device=self.device)
        rew = torch.tensor(episode.rewards[:T0], dtype=torch.float32, device=self.device)
        done = torch.tensor(episode.terminations[:T0], dtype=torch.bool, device=self.device)

        rtg = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
        prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
        timesteps = torch.arange(T0, dtype=torch.long, device=self.device).unsqueeze(-1)

        if start == -1:
            pad_len = self.context_len - T0
            return {
                "observations": F.pad(observations, (0, 0, pad_len, 0)),
                "actions": F.pad(actions, (0, 0, pad_len, 0)),
                "reward": F.pad(rew.unsqueeze(-1), (0, 0, pad_len, 0)),
                "done": F.pad(done.unsqueeze(-1), (0, 0, pad_len, 0)),
                "return_to_go": F.pad(rtg, (0, 0, pad_len, 0)),
                "prev_actions": F.pad(prev_act, (0, 0, pad_len, 0)),
                "timesteps": F.pad(timesteps, (0, 0, pad_len, 0)),
                "dataset_name": dataset_name,
                "total_rewards": rew.sum().item()
            }
        else:
            end = start + self.context_len
            return {
                "observations": observations[start:end],
                "actions": actions[start:end],
                "reward": rew[start:end].unsqueeze(-1),
                "done": done[start:end].unsqueeze(-1),
                "return_to_go": rtg[start:end],
                "prev_actions": prev_act[start:end],
                "timesteps": timesteps[start:end],
                "dataset_name": dataset_name,
                "total_rewards": rew[start:end].sum().item()
            }

class StratifiedSampler(Sampler):
    def __init__(self, dataset: MinariTrajectoryDataset, N_per_class=None):
        self.dataset = dataset
        self.class_to_indices = {}
        for idx, (dataset_name, _, _) in enumerate(dataset.slice_infos):
            self.class_to_indices.setdefault(dataset_name, []).append(idx)

        self.N_per_class = N_per_class or min(len(v) for v in self.class_to_indices.values())

        self.indices = []
        for k, idxs in self.class_to_indices.items():
            chosen = random.sample(idxs, self.N_per_class)
            self.indices.extend(chosen)

        random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
# Prepare data for PCA
all_h_sequences = []
dataset_names = []
total_rewards = []

# Generate h_sequence for each trajectory
dataset = MinariTrajectoryDataset(datasets, context_len=150, device=device)
sampler = StratifiedSampler(dataset, N_per_class=10000)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)

for batch in tqdm(loader,total = len(loader)):
    timesteps = batch["timesteps"].squeeze(-1).to(device)
    states = batch["observations"].to(device)
    actions = batch["prev_actions"].to(device)
    returns_to_go = batch["return_to_go"].to(device)

    # Generate h_sequence
    h_sequences, _ = encoder(timesteps=timesteps, states=states, actions=actions, returns_to_go = returns_to_go)

    # Append h_sequence, dataset_name, and total_rewards
    for i,h_sequence in enumerate(h_sequences):
        all_h_sequences.append(h_sequence.detach().cpu().numpy().flatten())  # Flatten to 1D
        dataset_names.append(batch["dataset_name"][i])
        total_rewards.append(batch["total_rewards"][i])

# Apply PCA
total_rewards = [r.item() if hasattr(r, "item") else r for r in total_rewards]

# 2. 简化 dataset 名称
simplified_names = [name.split("/")[-1].replace("-dense-v2", "") for name in dataset_names]

# 3. PCA 降维
pca = PCA(n_components=3)
reduced = pca.fit_transform(np.array(all_h_sequences))  # shape [N, 3]

# 2. 准备数据
total_rewards = [r.item() if hasattr(r, "item") else r for r in total_rewards]
simplified_names = [name.split("/")[-1].replace("-dense-v2", "") for name in dataset_names]

df = pd.DataFrame({
    "PCA1": reduced[:, 0],
    "PCA2": reduced[:, 1],
    "PCA3": reduced[:, 2],
    "reward": total_rewards,
    "dataset": simplified_names
})

# 3. 绘制交互式 3D 散点图
fig = px.scatter_3d(
    df,
    x="PCA1",
    y="PCA2",
    z="PCA3",
    color="dataset",
    size="reward",
    hover_data=["dataset", "reward"],
    title="3D PCA of h_sequence (Interactive)"
)

fig.write_html("pca_plot.html")