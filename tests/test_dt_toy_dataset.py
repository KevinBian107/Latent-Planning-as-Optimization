import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.optim as optim
from src.models.decision_transformer import DecisionTransformer  # 假设你的DT在 src/models/DT.py

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

def train_decision_transformer():
    batch_size = 128
    context_len = 8
    state_dim = 17
    act_dim = 6

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=2,
        h_dim=64,
        context_len=context_len,
        n_heads=2,
        drop_p=0.1,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    pbar = tqdm(range(3000))

    # 记录loss
    total_loss_list = []
    action_loss_list = []
    return_loss_list = []

    for step in pbar:
        model.train()
        states, actions, returns_to_go = generate_simple_data(batch_size, context_len, state_dim, act_dim)
        states, actions, returns_to_go = states.to(device), actions.to(device), returns_to_go.to(device)

        timesteps = torch.randint(0, 4096, (batch_size, context_len)).to(device)

        state_preds, action_preds, return_preds = model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go
        )

        # 🔵 Decision Transformer标准做法：对整个序列都做监督
        # actions的loss (action_preds vs actions)，shape (batch, context_len, act_dim)
        loss_action = torch.mean((action_preds - actions) ** 2)

        # return-to-go的loss (return_preds vs returns_to_go)，shape (batch, context_len, 1)
        loss_return = torch.mean((return_preds - returns_to_go) ** 2)

        # 总loss（动作loss为主，reward loss次要，可以加个小权重，比如0.25）
        total_loss = loss_action + 0.25 * loss_return

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_list.append(total_loss.item())
        action_loss_list.append(loss_action.item())
        return_loss_list.append(loss_return.item())

        if step % 10 == 0:
            pbar.set_description(f"Step {step}: total_loss={total_loss.item():.6f}, loss_action={loss_action.item():.6f}, loss_return={loss_return.item():.6f}")

    print("Finished training!")

    # 绘制loss曲线
    plt.figure(figsize=(10,6))
    plt.plot(total_loss_list, label='Total Loss')
    plt.plot(action_loss_list, label='Action Prediction Loss')
    plt.plot(return_loss_list, label='Return Prediction Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Decision Transformer Training Loss Curve (Standard)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_decision_transformer()
