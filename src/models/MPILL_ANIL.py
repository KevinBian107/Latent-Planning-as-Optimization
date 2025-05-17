import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.unet1d import Unet1D
from src.models.conditional_decision_transformer import ConditionalDecisionTransformer
from src.models.ANILGenerator import ANILGenerator
import pdb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class InferenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, hidden_dim) 
        self.logstd = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mu(h)
        std = torch.exp(torch.clamp(self.logstd(h), -10, 2))
        return mu, std

class LLDecoder(nn.Module):
    def __init__(self, zeta_dim: int, alpha_dim: int, hidden_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(zeta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, alpha_dim)
        self.logstd = nn.Linear(hidden_dim, alpha_dim)

    def forward(self, zeta: torch.Tensor):
        h = self.decoder(zeta)
        mu_alpha = self.mu(h)
        std_alpha = torch.exp(torch.clamp(self.logstd(h), -10, 2))
        return mu_alpha, std_alpha



class MPILL_ANIL(nn.Module):
    def __init__(self, state_dim, act_dim, context_len, h_dim=128, n_blocks=4, n_heads=2,
                 drop_p=0.1, n_latent=4, device=None, z_n_iters=3, langevin_step_size=0.3, noise_factor=1.0):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_len = context_len
        self.h_dim = h_dim
        self.n_latent = n_latent
        self.z_dim = h_dim * n_latent
        self.alpha_dim = h_dim

        input_size = (state_dim + act_dim + 1) * context_len

        self.ll_encoder = InferenceEncoder(input_size, hidden_dim=h_dim).to(self.device)
        self.ll_decoder = LLDecoder(zeta_dim=h_dim, alpha_dim=self.alpha_dim, hidden_dim=h_dim).to(self.device)
        assert self.alpha_dim % 2 == 0, "alpha_dim must be divisible by 2 for ANIL-style chunking"
        self.generator = ANILGenerator(
            z_dim=self.z_dim,
            alpha_feat_dim=self.alpha_dim // 2,
            alpha_head_dim=self.alpha_dim // 2
        ).to(self.device)
        self.generator.freeze_feature_extractor()
        self.trajectory_generator = ConditionalDecisionTransformer(
            state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(self.device)
        self.reward_head = nn.Sequential(
            nn.Linear(self.z_dim, 256), Swish(), nn.Linear(256, 1)).to(self.device)

        self.z_n_iters = z_n_iters
        self.step_size = langevin_step_size
        self.noise_factor = noise_factor

    def infer_alpha(self, states, actions, rewards):
        B, T, _ = states.shape
        returns = rewards[:, :, -1:]
        traj = torch.cat([states, actions, returns], dim=-1).reshape(B, -1)
        mu_zeta, std_zeta = self.ll_encoder(traj)
        zeta = mu_zeta + std_zeta * torch.randn_like(std_zeta)
        mu_alpha, std_alpha = self.ll_decoder(zeta)
        alpha_0 = mu_alpha + std_alpha * torch.randn_like(std_alpha)
        return alpha_0, mu_zeta, std_zeta, mu_alpha, zeta

    def langevin_refine(self, z0, states, actions, rewards, timesteps, alpha):
        for _ in range(self.z_n_iters):
            z0 = z0.detach().clone().requires_grad_(True)
            if hasattr(self.generator, 'forward') and 'alpha_head' in self.generator.forward.__code__.co_varnames:
                alpha_feat, alpha_head = alpha.chunk(2, dim=-1)
                z = self.generator(z0, alpha_feat=alpha_feat, alpha_head=alpha_head, requires_grad=True) #[B,z_dim]
                z_reshaped = z.view(z0.size(0), self.n_latent, self.h_dim) # 【B，n_latent, latent_dim]
                # z_dim == n_latent × latent_dim

            else:
                z = self.generator(z0, alpha)
                z_reshaped = z.view(z0.size(0), self.n_latent, self.alpha_dim)

            pred_reward = self.reward_head(z).squeeze(-1)
            returns = rewards[:, -1, 0]
            reward_loss = F.mse_loss(pred_reward, returns)

            pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped)
            action_loss = F.mse_loss(pred_action, actions[:, -1, :])
            total_loss = reward_loss + action_loss

            grad = torch.autograd.grad(total_loss, z0)[0]
            z0 = z0 - 0.5 * self.step_size ** 2 * grad
            z0 += self.noise_factor * self.step_size * torch.randn_like(z0)

        return z0.detach()

    def forward(self, states, actions, rewards, timesteps,
                alpha_bar: Optional[torch.Tensor] = None,
                compute_loss: bool = True):
        alpha_0, mu_zeta, std_zeta, mu_alpha, zeta = self.infer_alpha(states, actions, rewards)
        z0 = torch.randn(states.size(0), self.z_dim, device=self.device)
        z0 = self.langevin_refine(z0, states, actions, rewards, timesteps, alpha_0)
        if hasattr(self.generator, 'forward') and 'alpha_head' in self.generator.forward.__code__.co_varnames:
            alpha_feat_0, alpha_head_0 = alpha_0.chunk(2, dim=-1)
            z = self.generator(z0, alpha_feat=alpha_feat_0, alpha_head=alpha_head_0)
            z_reshaped = z.view(z0.size(0), self.n_latent, self.h_dim)

        else:
            z = self.generator(z0, alpha_0)
            z_reshaped = z.view(z0.size(0), self.n_latent, self.alpha_dim)

        if not compute_loss:
            pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped)
            pred_reward = self.reward_head(z).squeeze(-1)
            return pred_action, pred_reward, alpha_0, mu_alpha, zeta

        if alpha_bar is not None:
            if alpha_bar.shape != mu_alpha.shape:
                if len(alpha_bar.shape) == 1 and len(mu_alpha.shape) == 2:
                    alpha_bar_expanded = alpha_bar.unsqueeze(0).expand(mu_alpha.size(0), -1)
                    alpha_loss = F.mse_loss(mu_alpha, alpha_bar_expanded)
                else:
                    alpha_loss = torch.tensor(0.0, device=self.device)
            else:
                alpha_loss = F.mse_loss(mu_alpha, alpha_bar)
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped)
        pred_reward = self.reward_head(z).squeeze(-1)
        kl_zeta = -0.5 * torch.sum(1 + torch.log(std_zeta**2 + 1e-8) - mu_zeta**2 - std_zeta**2, dim=1).mean()

        return pred_action, pred_reward, alpha_0, alpha_loss, kl_zeta, mu_alpha, zeta
