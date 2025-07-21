import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,IncrementalPCA
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import pandas as pd
import plotly.express as px
import random
import minari
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Sampler
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from agent.src.models.lpt_experiment import DecisionTransformerEncoder
import plotly.graph_objects as go
# Load the trained encoder model
###
device = torch.device("cpu")
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
    


# Generate h_sequence for each trajectory
dataset = MinariTrajectoryDataset(datasets, context_len=150, device=device)
sampler = StratifiedSampler(dataset, N_per_class=5000)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1)

from torch.utils.data import DataLoader
import torch

all_h_sequences = []
dataset_names = []
total_rewards = []
desired_goal_sequences = [] 


for batch in tqdm(loader, desc="Processing trajectories"):
    rtg = batch["return_to_go"].squeeze(0)     
    obs = batch["observations"].squeeze(0)     
    act = batch["actions"].squeeze(0)           

    # 拼接： [context_len, d_r + d_s + d_a]
    h_seq = torch.cat([obs, act], dim=-1)   
    all_h_sequences.append(h_seq)
    desired_goal = obs[:, 6 : 8]  # [T, 2]
    desired_goal_sequences.append(desired_goal)

    dataset_names.append(batch["dataset_name"][0])
    total_rewards.append(batch["total_rewards"][0].item())


# Apply PCA
all_h_sequences = np.array([i.numpy() for i in all_h_sequences]).reshape(len(all_h_sequences), -1)

pca = PCA(n_components=3)
reduced = pca.fit_transform(np.array(all_h_sequences))  # shape [N, 3]
total_rewards = [r.item() if hasattr(r, "item") else r for r in total_rewards]
simplified_names = [name.split("/")[-1].replace("-dense-v2", "") for name in dataset_names]
index = np.arange(len(simplified_names))
df = pd.DataFrame({
    "PCA1": reduced[:, 0],
    "PCA2": reduced[:, 1],
    "PCA3": reduced[:, 2],
    "reward": total_rewards,
    "dataset": simplified_names,
    "index":index
})

import dash
from dash import dcc, html, Input, Output
# ==== Dash 页面 ====
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id="pca-3d",
            figure=px.scatter_3d(
                df,
                x="PCA1", y="PCA2", z="PCA3",
                opacity=0.7,
                color="dataset",
                size="reward",
                hover_data=["dataset", "reward", "index"],
                custom_data=["index"],  # 👈 非常关键：hover/click 可读取 index
                title="3D PCA of h_sequence (Interactive)"
            ).update_traces(marker=dict(size=5))
        )
    ], style={"height": "800px","width": "65%", "display": "inline-block", "verticalAlign": "top"}), 

    html.Div([
        dcc.Graph(id="goal-trajectory")
    ], style={"width": "34%", "display": "inline-block", "verticalAlign": "top", "paddingLeft": "1%"})  # 👈 控制间距
])

# ==== 回调函数：hover 时绘图 ====
@app.callback(
    Output("goal-trajectory", "figure"),
    Input("pca-3d", "hoverData")
)
def show_trajectory_on_hover(hoverData):
    if hoverData is None:
        return go.Figure()

    traj_index = hoverData["points"][0]["customdata"][0]
    traj = desired_goal_sequences[traj_index].numpy()
    num_steps = traj.shape[0]

    # 步长归一化，用于颜色渐变（0.0 - 1.0）
    color_steps = np.linspace(0, 1, num_steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traj[:, 0],
        y=traj[:, 1],
        mode="lines+markers",
        marker=dict(
            size=6,
            color=color_steps,              # 每个点颜色表示时间
            colorscale="Bluered",           # 从蓝到红
            colorbar=dict(title="Step"),    # 显示颜色条
            showscale=True
        ),
        line=dict(
            color='rgba(0,0,0,0)',          # 让线不覆盖点颜色（可选）
            width=1
        ),
        name=f"Trajectory {traj_index}"
    ))

    fig.update_layout(
        title=f"Desired Goal Trajectory #{traj_index}",
        xaxis_title="goal_x",
        yaxis_title="goal_y",
        height=500
    )
    return fig


app.run(debug=True)