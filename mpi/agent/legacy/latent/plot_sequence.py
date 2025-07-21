import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_desired_goal_sequences(
    desired_goal_sequences,
    rewards=None,
    max_trajs=100,
    show_index=False,
    figsize=(8, 6),
    alpha=0.6,
    linewidth=1.5,
    title="Desired Goal Sequences"
):
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap("viridis")
    num_trajs = min(len(desired_goal_sequences), max_trajs)

    for i in range(num_trajs):
        traj = desired_goal_sequences[i]
        if isinstance(traj, torch.Tensor):
            traj = traj.cpu().numpy()

        xs = traj[:, 0]
        ys = traj[:, 1]

        color = cmap(i / num_trajs) if rewards is None else cmap(rewards[i] / max(rewards))
        plt.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)

        if show_index:
            plt.text(xs[-1], ys[-1], str(i), fontsize=6, alpha=0.8)

    plt.xlabel("desired_goal_x")
    plt.ylabel("desired_goal_y")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 👇 构造多个不同的轨迹
def create_spiral_trajectory(T=100, a=0.1, b=0.15, phase=0.0):
    theta = np.linspace(0, 4 * np.pi, T)
    r = a + b * theta
    x = r * np.cos(theta + phase)
    y = r * np.sin(theta + phase)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

def create_random_walk_trajectory(T=100, step_size=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.randn(T, 2) * step_size
    traj = np.cumsum(steps, axis=0)
    return torch.tensor(traj, dtype=torch.float32)

if  __name__ == "__main__":
    # ✅ 构建多个轨迹
    spiral_trajs = [create_spiral_trajectory(phase=np.random.uniform(0, np.pi)) for _ in range(5)]
    random_trajs = [create_random_walk_trajectory(seed=i) for i in range(5)]

    # ✅ 合并并绘图
    all_trajs = spiral_trajs + random_trajs

    plot_desired_goal_sequences(
        desired_goal_sequences=all_trajs,
        show_index=True,
        title="Multiple Desired Goal Trajectories (Spiral + Random Walk)"
    )
