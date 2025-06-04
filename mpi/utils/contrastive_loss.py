import torch
import torch.nn.functional as F
def reward_latent_consistency_loss(z_latent, pred_reward):
    """
    z_latent: [B, T, D] (e.g., 128, 8, 64)
    pred_reward: [B]

    Returns:
        Scalar loss enforcing that similar predicted rewards have similar latent.
    """
    B, T, D = z_latent.shape

    # Step 1: Pool latent over time (mean-pool)
    z_flat = z_latent.mean(dim=1)  # [B, D]

    # Step 2: Normalize latent vectors
    z_norm = F.normalize(z_flat, dim=-1)  # [B, D]

    # Step 3: Compute latent similarity matrix
    sim_latent = torch.matmul(z_norm, z_norm.T)  # [B, B]

    # Step 4: Compute reward similarity matrix
    pred_reward = pred_reward.unsqueeze(1)  # [B, 1]
    reward_diff = torch.abs(pred_reward - pred_reward.T)  # [B, B]
    reward_sim = 1.0 - reward_diff / (reward_diff.max() + 1e-8)  # normalize to [0, 1]

    # Optional: mask diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=z_latent.device)
    sim_latent_flat = sim_latent[mask]
    reward_sim_flat = reward_sim[mask]

    # Step 5: Compute MSE loss between similarity patterns
    loss = F.mse_loss(sim_latent_flat, reward_sim_flat)

    return loss