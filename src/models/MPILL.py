import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.unet1d import Unet1D
from src.models.conditional_decision_transformer import ConditionalDecisionTransformer
from src.models.alpha_networks import MLPGenerator
import pdb

class Swish(nn.Module):
    """Simple Swish activation function"""
    def forward(self, x):
        """
        Used for the reward models
        
        Args:
            x: Input with shape [B, *]
        
        Return:
            Applied x * sigmoid(x) with same shape [B, *] as input
        """
        
        return x * torch.sigmoid(x)


class InferenceEncoder(nn.Module):
    """Encoder network for variational inference"""
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
        """
        Forward pass of encoder
        
        Args:
            x: Input tensor [B, input_dim]
            
        Returns:
            mu: Mean of distribution [B, hidden_dim]
            std: Standard deviation of distribution [B, hidden_dim]
        """
        h = self.encoder(x) 
        mu = self.mu(h) 
        std = torch.exp(torch.clamp(self.logstd(h), -10, 2))
        return mu, std


class LLDecoder(nn.Module):
    """Decoder network for latent plan generation"""
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
        """
        Forward pass of decoder
        
        Args:
            zeta: Latent code [B, zeta_dim]
            
        Returns:
            mu_alpha: Mean of alpha distribution [B, alpha_dim]
            std_alpha: Standard deviation of alpha distribution [B, alpha_dim]
        """
        h = self.decoder(zeta)
        mu_alpha = self.mu(h)
        std_alpha = torch.exp(torch.clamp(self.logstd(h), -10, 2))
        return mu_alpha, std_alpha


class MPILearningLearner(nn.Module):
    """Meta-Planning as Inference Learning Learner (TD-BU-Clean algorithm)"""
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

        # Input size for encoder: flattened trajectory
        input_size = (state_dim + act_dim + 1) * context_len
        
        # Learning Learner encoder: q_φ(ζ | τ, y)
        # Input: flattened trajectory [B, input_size]
        # Output: parameters for distribution over ζ
        self.ll_encoder = InferenceEncoder(input_size, hidden_dim=h_dim).to(self.device)
        
        # Learning Learner decoder: G_ψ(ζ) → μ_α, σ_α
        # Input: latent code ζ [B, h_dim]
        # Output: parameters for distribution over α
        self.ll_decoder = LLDecoder(zeta_dim=h_dim, alpha_dim=self.alpha_dim, hidden_dim=h_dim).to(self.device)
        
        # Generator conditioned on α₀
        # Input: random z0 [B, z_dim] and α [B, alpha_dim]
        # Output: latent plan z [B, z_dim]
        self.generator = MLPGenerator(z_dim=self.z_dim, alpha_dim=self.alpha_dim).to(self.device)
        
        # Trajectory generator
        # Input: states [B, T, state_dim], actions [B, T, act_dim], timesteps [B, T], latent code z [B, n_latent, h_dim]
        # Output: predicted next action [B, act_dim]
        self.trajectory_generator = ConditionalDecisionTransformer(
            state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(self.device)
        
        # Reward prediction head
        # Input: latent code z [B, z_dim]
        # Output: predicted reward [B, 1]
        self.reward_head = nn.Sequential(
            nn.Linear(self.z_dim, 256), 
            Swish(), 
            nn.Linear(256, 1)
        ).to(self.device)

        self.z_n_iters = z_n_iters
        self.step_size = langevin_step_size
        self.noise_factor = noise_factor

    def infer_alpha(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        """
        Infer alpha from trajectory using Learning Learner encoder and decoder
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            rewards: Reward sequence [B, T, 1]
            
        Returns:
            alpha_0: Sampled alpha [B, alpha_dim]
            mu_zeta: Mean of zeta distribution [B, h_dim]
            std_zeta: Standard deviation of zeta distribution [B, h_dim]
            mu_alpha: Mean of alpha distribution [B, alpha_dim]
            zeta: Sampled zeta [B, h_dim]
        """
        B, T, _ = states.shape
        
        # returns = torch.sum(rewards, dim=2).unsqueeze(-1)
        returns = rewards[:,:,-1:]
        # pdb.set_trace()
        
        traj = torch.cat([states, actions, returns], dim=-1)  # [B, T, state_dim+act_dim+1]
        traj = traj.reshape(B, -1)  # [B, T*(state_dim+act_dim+1)]

        mu_zeta, std_zeta = self.ll_encoder(traj) # Both: [B, h_dim]
        zeta = mu_zeta + std_zeta * torch.randn_like(std_zeta) # [B, h_dim]
        mu_alpha, std_alpha = self.ll_decoder(zeta) # Both: [B, alpha_dim]
        alpha_0 = mu_alpha + std_alpha * torch.randn_like(std_alpha) # [B, alpha_dim]
        
        return alpha_0, mu_zeta, std_zeta, mu_alpha, zeta

    def langevin_refine(self, z0: torch.Tensor, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, timesteps: torch.Tensor, alpha_0: torch.Tensor):
        """
        Refine z0 using Langevin dynamics
        
        Args:
            z0: Initial random latent code [B, z_dim]
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            rewards: Reward sequence [B, T, 1]
            timesteps: Timestep indices [B, T]
            alpha_0: Conditioning parameter [B, alpha_dim]
            
        Returns:
            Refined z0 [B, z_dim]
        """
        for _ in range(self.z_n_iters):
            z0 = z0.detach().clone().requires_grad_(True) # [B, z_dim]
            
            z = self.generator(z0, alpha_0) # [B, z_dim]
            z_reshaped = z.view(z0.size(0), self.n_latent, self.alpha_dim) # [B, n_latent, alpha_dim]
            
            pred_reward = self.reward_head(z).squeeze(-1) # [B]
            returns = rewards[:,:,-1:] #torch.sum(rewards, dim=2).unsqueeze(-1)
            reward_loss = F.mse_loss(pred_reward, returns) # Scalar
            
            pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped) # [B, act_dim]
            action_loss = F.mse_loss(pred_action, actions[:, -1, :]) # Scalar
            
            total_loss = reward_loss + action_loss # Scalar

            grad = torch.autograd.grad(total_loss, z0)[0] # [B, z_dim]
            
            # peform Langevin update
            z0 = z0 - 0.5 * self.step_size ** 2 * grad # [B, z_dim]
            
            # add noise
            z0 += self.noise_factor * self.step_size * torch.randn_like(z0)  # [B, z_dim]

        return z0.detach() # [B, z_dim]

    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                timesteps: torch.Tensor,
                alpha_bar: Optional[torch.Tensor] = None,
                compute_loss: bool = True):
        """
        Forward pass of Meta-Planning as Inference model
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            rewards: Reward sequence [B, T, 1]
            timesteps: Timestep indices [B, T]
            alpha_bar: Optional supervision signal for alpha [B, alpha_dim]
            compute_loss: Whether to compute losses
            
        Returns:
            If compute_loss=True:
                pred_action: Predicted action [B, act_dim]
                pred_reward: Predicted reward [B]
                alpha_0: Sampled alpha [B, alpha_dim]
                alpha_loss: Alpha supervision loss (scalar)
                kl_zeta: KL divergence loss for zeta (scalar)
                mu_alpha: Mean of alpha distribution [B, alpha_dim]
                zeta: Sampled zeta [B, h_dim]
            If compute_loss=False:
                pred_action: Predicted action [B, act_dim]
                pred_reward: Predicted reward [B]
                alpha_0: Sampled alpha [B, alpha_dim]
                mu_alpha: Mean of alpha distribution [B, alpha_dim]
                zeta: Sampled zeta [B, h_dim]
        """
        # Infer alpha
        alpha_0, mu_zeta, std_zeta, mu_alpha, zeta = self.infer_alpha(states, actions, rewards)
        
        # sample initial z0 (gaussian) and refine with Langevin dynamics
        z0 = torch.randn(states.size(0), self.z_dim, device=self.device) # [B, z_dim]
        z0 = self.langevin_refine(z0, states, actions, rewards, timesteps, alpha_0) # [B, z_dim]
        
        # generate z from refined z0
        z = self.generator(z0, alpha_0) # [B, z_dim]
        z_reshaped = z.view(z0.size(0), self.n_latent, self.alpha_dim) # [B, n_latent, alpha_dim]
        
        if not compute_loss:
            pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped)  # [B, act_dim]
            pred_reward = self.reward_head(z).squeeze(-1)  # [B]
            return pred_action, pred_reward, alpha_0, mu_alpha, zeta
        
        # bottom-up supervision loss
        if alpha_bar is not None:
            if alpha_bar.shape != mu_alpha.shape:
                # If alpha_bar doesn't have batch dimension, expand it
                if len(alpha_bar.shape) == 1 and len(mu_alpha.shape) == 2:
                    # alpha_bar: [alpha_dim] -> [B, alpha_dim]
                    alpha_bar_expanded = alpha_bar.unsqueeze(0).expand(mu_alpha.size(0), -1)
                    alpha_loss = F.mse_loss(mu_alpha, alpha_bar_expanded)
                else:
                    print(f"Incompatible shapes: alpha_bar {alpha_bar.shape}, mu_alpha {mu_alpha.shape}")
                    alpha_loss = torch.tensor(0.0, device=self.device)
            else:
                alpha_loss = F.mse_loss(mu_alpha, alpha_bar)
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        pred_action, _ = self.trajectory_generator(timesteps, states, actions, z_reshaped) # [B, act_dim]
        pred_reward = self.reward_head(z).squeeze(-1)  # [B]
        kl_zeta = -0.5 * torch.sum(1 + torch.log(std_zeta**2 + 1e-8) - mu_zeta**2 - std_zeta**2, dim=1).mean() # Scalar
        
        return pred_action, pred_reward, alpha_0, alpha_loss, kl_zeta, mu_alpha, zeta
