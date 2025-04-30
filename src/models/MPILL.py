import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.unet1d import Unet1D
from src.models.conditional_decision_transformer import ConditionalDecisionTransformer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int, alpha_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.alpha_dim = alpha_dim

        # MLP that maps z0 → z, conditioned on α via FiLM
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)

        # FiLM modulation from α
        self.film1 = nn.Linear(alpha_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(alpha_dim, 2 * z_dim)

        self.activation = nn.ReLU()

    def apply_film(self, x: torch.Tensor, film: nn.Linear, alpha: torch.Tensor) -> torch.Tensor:
        scale_shift = film(alpha).unsqueeze(-1) if x.dim() == 3 else film(alpha)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1 + scale) + shift

    def forward(self, z0: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        z0: [B, z_dim]
        alpha: [B, alpha_dim]
        returns: [B, z_dim]
        """
        x = self.fc1(z0)
        x = self.apply_film(x, self.film1, alpha)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.apply_film(x, self.film2, alpha)
        return x


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
        return self.mu(h), torch.exp(self.logstd(h))


class MPILearningLearner(nn.Module):
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
                 z_n_iters: int = 3,
                 langevin_step_size: float = 0.3,
                 noise_factor: float = 1.0):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_len = context_len
        self.h_dim = h_dim
        self.n_latent = n_latent
        self.z_dim = h_dim * n_latent
        self.alpha_dim = h_dim

        input_size = (state_dim + act_dim + 1) * context_len
        
        # LL encoder qϕ(ζ | τ, y)
        self.ll_encoder = InferenceEncoder(input_size, hidden_dim=h_dim).to(self.device)
        
        # LL decoder Gψ(ζ) → μᵅ, σᵅ
        self.ll_decoder = InferenceEncoder(h_dim, hidden_dim=h_dim).to(self.device)
        
        # Generator conditioned on α₀
        # self.generator = FiLMModulatedUnet1D(dim=self.z_dim, channels=1, dim_mults=(1, 2, 4)).to(self.device)
        self.generator = MLPGenerator(z_dim=self.z_dim, alpha_dim=self.alpha_dim).to(self.device)

        
        # Trajectory & reward heads
        self.trajectory_generator = ConditionalDecisionTransformer(
            state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(self.device)
        self.reward_head = nn.Sequential(
            nn.Linear(self.z_dim, 256), Swish(), nn.Linear(256, 1)
        ).to(self.device)

        self.z_n_iters = z_n_iters
        self.step_size = langevin_step_size
        self.noise_factor = noise_factor

    def infer_alpha(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        B, T, _ = states.shape
        returns = rewards[:, -1:].expand(-1, T, 1)
        traj = torch.cat([states, actions, returns], dim=-1).reshape(B, -1)

        mu_zeta, std_zeta = self.ll_encoder(traj)
        zeta = mu_zeta + std_zeta * torch.randn_like(std_zeta)

        mu_alpha, std_alpha = self.ll_decoder(zeta)
        alpha_0 = mu_alpha + std_alpha * torch.randn_like(std_alpha)
        return alpha_0, mu_zeta, std_zeta, mu_alpha

    def langevin_refine(self, z0: torch.Tensor, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, timesteps: torch.Tensor, alpha_0: torch.Tensor):
        for _ in range(self.z_n_iters):
            z0 = z0.detach().clone().requires_grad_(True)
            # z = self.generator(z0.view(z0.size(0), 1, -1)).view(z0.size(0), self.n_latent, self.h_dim)
            z = self.generator(z0, alpha_0).view(z0.size(0), self.n_latent, self.alpha_dim)
            
            pred_reward = self.reward_head(z.view(z.size(0), -1)).squeeze(-1)
            reward_loss = F.mse_loss(pred_reward, rewards[:, -1, 0])

            pred_action, _ = self.trajectory_generator(timesteps, states, actions, z)
            action_loss = F.mse_loss(pred_action, actions[:, -1, :])
            total_loss = reward_loss + action_loss

            grad = torch.autograd.grad(total_loss, z0)[0]
            z0 = z0 - 0.5 * self.step_size ** 2 * grad
            z0 += self.noise_factor * self.step_size * torch.randn_like(z0)

        return z0.detach()

    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                timesteps: torch.Tensor,
                alpha_bar: Optional[torch.Tensor] = None):
        
        alpha_0, mu_zeta, std_zeta, mu_alpha = self.infer_alpha(states, actions, rewards)
        z0 = torch.randn(states.size(0), self.z_dim, device=self.device)
        z0 = self.langevin_refine(z0, states, actions, rewards, timesteps, alpha_0)
        # z = self.generator(z0.view(z0.size(0), 1, -1)).view(z0.size(0), self.n_latent, self.h_dim)
        z = self.generator(z0, alpha_0).view(z0.size(0), self.n_latent, self.alpha_dim)

        
        # Bottom-up supervision loss
        if alpha_bar is not None and alpha_bar.shape == mu_alpha.shape:
            alpha_loss = F.mse_loss(mu_alpha, alpha_bar.expand_as(mu_alpha))
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        pred_action, _ = self.trajectory_generator(timesteps, states, actions, z)
        pred_reward = self.reward_head(z.view(z.size(0), -1)).squeeze(-1)

        kl_zeta = -0.5 * torch.sum(1 + torch.log(std_zeta**2 + 1e-8) - mu_zeta**2 - std_zeta**2, dim=1).mean()
        return pred_action, pred_reward, alpha_0, alpha_loss, kl_zeta
