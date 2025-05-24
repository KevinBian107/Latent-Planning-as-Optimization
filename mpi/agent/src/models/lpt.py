import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.src.models.conditional_decision_transformer import ConditionalDecisionTransformer
from agent.src.models.unet1d import Unet1D
from agent.src.models.vae1d import VAE1D
from agent.src.util_function import register_model

from typing import Optional, Tuple

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
@register_model("BasicLPT")
class LatentPlannerModel(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 act_dim: int, 
                 context_len: int, 
                 h_dim: int = 128, 
                 n_blocks: int = 4, 
                 n_heads: int = 2, 
                 drop_p: float = 0.1, 
                 n_latent: int = 4,
                 device: Optional[torch.device] = None,
                 reward_weight: float = 1.0,
                 action_weight: float = 1.0,
                 z_n_iters: int = 3,
                 langevin_step_size: float = 0.3,
                 noise_factor: float = 1.0):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len
        self.n_latent = n_latent
        self.z_dim = h_dim * n_latent
        self.z_best_idx = 0
        self.y_max = 0

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.planner = Unet1D(dim=self.z_dim, channels=1, dim_mults=(1, 2, 4)).to(self.device)
        self.planner = VAE1D(dim=self.z_dim).to(self.device)
        
        self.trajectory_generator = ConditionalDecisionTransformer(
            state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p
        ).to(self.device)
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            Swish(),
            nn.Linear(256, 1)
        ).to(self.device)

        # Langevin sampling settings
        self.z_n_iters = z_n_iters
        self.z_with_noise = True
        self.noise_factor = noise_factor
        self.langevin_step_size = langevin_step_size

        # Loss weights
        self.reward_weight = reward_weight
        self.action_weight = action_weight

        # z buffer
        self.z_buffer = torch.randn(1000, self.n_latent, self.h_dim, device=self.device)

    def unet_forward(self, z):
        B = z.size(0)
        z = z.view(B, 1, -1)
        z = self.planner(z)
        z = z.view(B, self.n_latent, self.h_dim)
        return z

    def reward_forward(self, z):
        z = z.view(z.size(0), -1)
        return self.reward_predictor(z).squeeze(-1)

    def action_forward(self, states, actions, timesteps, z_latent):
        pred_actions, _ = self.trajectory_generator(timesteps, states, actions, z_latent)
        return pred_actions
    
    def infer_z_given_y(self, y, step_size=0.3, debug=False, omega_cg=1.0):
        self.eval()

        batch_size = y.shape[0]
        device = y.device
        step_size = self.langevin_step_size

        # Use best-known z as initialization (or random)
        z = self.z_buffer[self.z_best_idx].unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)

        z_mse_grads_norm = []

        for i in range(self.z_n_iters):
            z = z.detach().clone()
            z.requires_grad_(True)

            z_unet = self.unet_forward(z)
            y_hat = self.reward_forward(z_unet)

            mse = F.mse_loss(y_hat, y.squeeze(), reduction='sum')
            z_grad = torch.autograd.grad(mse, z)[0].detach()

            # Langevin update
            z = z - 0.5 * step_size**2 * (z + z_grad * omega_cg)

            if self.z_with_noise:
                z += self.noise_factor * step_size * torch.randn_like(z)

            z_mse_grads_norm.append(torch.norm(z_grad, dim=1).mean().item())

            if debug:
                print(f"[{i}] MSE: {mse.item():.4f}, GradNorm: {z_mse_grads_norm[-1]:.4f}")

        z = z.detach()
        #FIXME if it should not change back to self.train
        #self.train()
        return z, z_mse_grads_norm

    def infer_z(self, states, actions, timesteps, rewards, batch_inds):
        self.eval()

        z = self.z_buffer[batch_inds]

        for _ in range(self.z_n_iters):
            z = z.detach().clone()
            z.requires_grad_(True)

            z_latent = self.unet_forward(z)

            # Predict reward
            pred_rewards = self.reward_forward(z_latent)
            reward_loss = torch.nn.MSELoss()(pred_rewards, rewards.squeeze(1))

            # Predict action
            pred_actions, _ = self.trajectory_generator(timesteps, states, actions, z_latent)
            target_actions = actions[:, -1, :] 
            action_loss = F.mse_loss(pred_actions, target_actions, reduction='mean')

            # Combine
            total_loss = self.reward_weight * reward_loss + self.action_weight * action_loss

            grad = torch.autograd.grad(total_loss, z)[0]
            z = z - 0.5 * self.langevin_step_size**2 * grad

            if self.z_with_noise:
                z += self.noise_factor * self.langevin_step_size * torch.randn_like(z)

        z = z.detach()
        self.z_buffer[batch_inds] = z
        self.train()
        return z

    def forward(self, 
                states: torch.Tensor, 
                actions: torch.Tensor, 
                timesteps: torch.Tensor, 
                rewards: torch.Tensor, 
                batch_inds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)
        rewards = rewards.to(self.device)
        batch_inds = batch_inds.to(self.device)

        # 1. Infer z
        if self.training:
            z = self.infer_z(states, actions, timesteps, rewards, batch_inds)
            max_y = torch.max(rewards)
            max_idx = torch.argmax(rewards)
            if max_y > self.y_max:
                self.y_max = max_y
                self.z_best_idx = max_idx 
        else:
            z,_ = self.infer_z_given_y(rewards,step_size=self.langevin_step_size)

        # 2. Refine z
        z_latent = self.unet_forward(z)

        # 3. Predict next action and state
        pred_action, pred_state = self.trajectory_generator(timesteps, states, actions, z_latent)

        # 4. Predict reward
        pred_reward = self.reward_forward(z_latent)

        return pred_action, pred_state, pred_reward