import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.optim as optim
from src.models.LPT import LatentPlannerModel

def generate_simple_data(batch_size, context_len, state_dim, act_dim):
    """
    用简单规则生成数据
    """
    # 时间步（用来制造state变化）
    t = torch.linspace(0, 1, context_len).unsqueeze(0).expand(batch_size, -1)  # (batch, context_len)

    # state: 线性增长 + 周期性成分
    states = []
    for i in range(state_dim):
        if i % 2 == 0:
            # 偶数维度：线性增长
            states.append(t + 0.1 * torch.randn_like(t))  
        else:
            # 奇数维度：sin波动
            states.append(torch.sin(2 * math.pi * t * (i+1)) + 0.1 * torch.randn_like(t))
    states = torch.stack(states, dim=-1)  # (batch, context_len, state_dim)
        # 动作: 取state前几维，线性加权求和
    weights = torch.linspace(1.0, 0.5, state_dim)[:act_dim]  # 简单的权重
    actions = states[:, :, :act_dim] * weights  # element-wise乘
    actions = actions.sum(dim=-1, keepdim=True)  # 求和，(batch, context_len, 1)
    actions = actions.expand(-1, -1, act_dim)  # 广播成 (batch, context_len, act_dim)

    # return_to_go: 设成动作的负norm，越小越好
    returns_to_go = -torch.norm(actions, dim=-1, keepdim=True)  # (batch, context_len, 1)

    return states, actions, returns_to_go

def train_latent_planner():
    batch_size = 128
    context_len = 8
    state_dim = 17
    act_dim = 6

    reward_weight = 0.25

    model = LatentPlannerModel(
        state_dim=state_dim,
        act_dim=act_dim,
        context_len=context_len,
        h_dim=32,
        n_blocks=2,
        n_heads=2,
        drop_p=0.1,
        n_latent=2,
        device=device,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pbar = tqdm(range(300))

    # 新增：记录 loss 的列表
    total_loss_list = []
    loss_ap_list = []
    loss_rp_list = []

    for step in pbar:
        model.train()
        states, actions, rewards = generate_simple_data(batch_size, context_len, state_dim, act_dim)
        rewards = rewards.squeeze(-1) 
        rewards = rewards[:, 0]
        states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
        timesteps = torch.randint(0, 1000, (batch_size, context_len)).to(device)
        batch_inds = torch.randint(0, 1000, (batch_size,)).to(device)

        pred_action, pred_state, pred_reward = model(states, actions, timesteps, rewards, batch_inds)

        action_targets = actions[:, -1, :]
        loss_ap = torch.mean((pred_action - action_targets) ** 2)
        rewards_targets = rewards
        loss_rp = torch.mean((pred_reward - rewards_targets) ** 2)

        total_loss = loss_ap + reward_weight * loss_rp

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 新增：每步记录loss
        total_loss_list.append(total_loss.item())
        loss_ap_list.append(loss_ap.item())
        loss_rp_list.append(loss_rp.item())

        if step % 10 == 0:
            pbar.set_description(f"Step {step}: total_loss={total_loss.item():.6f}, loss_ap={loss_ap.item():.6f}, loss_rp={loss_rp.item():.6f}")

    print("Finished training!")

    # 新增：绘制 loss 曲线
    plt.figure(figsize=(10,6))
    plt.plot(total_loss_list, label='Total Loss')
    plt.plot(loss_ap_list, label='Action Prediction Loss')
    plt.plot(loss_rp_list, label='Reward Prediction Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_latent_planner()
