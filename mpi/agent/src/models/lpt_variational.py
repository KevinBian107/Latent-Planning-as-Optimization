import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.conditional_decision_transformer import ConditionalDecisionTransformer
from src.models.unet1d import Unet1D
from typing import Optional, Tuple, Dict
from src.util_function import register_model

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class VariationalEncoder(nn.Module):
    """Encoder network q_φ(z|τ,y) that produces μ_φ(τ,y) and logσ_φ²(τ,y)"""
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Mean and log variance outputs
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of encoder
        
        Args:
            x: Input tensor [B, input_dim]
            
        Returns:
            mu: Mean of distribution [B, z_dim]
            logvar: Log variance of distribution [B, z_dim]
        """
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var)
        z = μ_φ + σ_φ ⊙ ε, ε ~ N(0,I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

@register_model("LPT_VI")
class LatentPlannerVIModel(nn.Module):
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
                 lr_alpha: float = 1e-4,  # Learning rate for p_α(z)
                 lr_beta: float = 1e-4,   # Learning rate for p_β(τ|z)
                 lr_gamma: float = 1e-4,  # Learning rate for p_γ(y|z)
                 lr_phi: float = 1e-4):   # Learning rate for q_φ(z|τ,y)
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len
        self.n_latent = n_latent
        self.z_dim = h_dim * n_latent

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder q_φ(z|τ,y)
        input_dim = (state_dim + act_dim + 1) * context_len  # Flattened trajectory
        self.encoder = VariationalEncoder(input_dim, h_dim, self.z_dim).to(self.device)

        # Decoder components
        # p_α(z) - prior distribution
        self.planner = Unet1D(dim=self.z_dim, channels=1, dim_mults=(1, 2, 4)).to(self.device)
        
        # p_β(τ|z) - trajectory generator
        self.trajectory_generator = ConditionalDecisionTransformer(
            state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p
        ).to(self.device)
        
        # p_γ(y|z) - reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            Swish(),
            nn.Linear(256, 1)
        ).to(self.device)

        # Loss weights
        self.reward_weight = reward_weight
        self.action_weight = action_weight

        # Learning rates
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.lr_gamma = lr_gamma
        self.lr_phi = lr_phi

        # Initialize optimizers for each component
        self.optimizer_alpha = torch.optim.Adam(self.planner.parameters(), lr=lr_alpha)
        self.optimizer_beta = torch.optim.Adam(self.trajectory_generator.parameters(), lr=lr_beta)
        self.optimizer_gamma = torch.optim.Adam(self.reward_predictor.parameters(), lr=lr_gamma)
        self.optimizer_phi = torch.optim.Adam(self.encoder.parameters(), lr=lr_phi)

    def encode(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory into latent distribution parameters
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            rewards: Reward sequence [B, T, 1]
            
        Returns:
            mu: Mean of latent distribution [B, z_dim]
            logvar: Log variance of latent distribution [B, z_dim]
        """
        B, T, _ = states.shape
        
        # Expand rewards to match sequence length
        returns = rewards[:, -1:].expand(-1, T, 1)
        
        # Flatten trajectory
        traj = torch.cat([states, actions, returns], dim=-1)  # [B, T, state_dim+act_dim+1]
        traj = traj.reshape(B, -1)  # [B, T*(state_dim+act_dim+1)]
        
        return self.encoder(traj)

    def decode(self, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent code into predictions
        
        Args:
            z: Latent code [B, z_dim]
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            timesteps: Timestep indices [B, T]
            
        Returns:
            pred_actions: Predicted actions [B, act_dim]
            pred_states: Predicted states [B, state_dim]
            pred_rewards: Predicted rewards [B]
        """
        # Process through planner
        z_latent = self.planner(z.view(-1, 1, self.z_dim)).view(-1, self.n_latent, self.h_dim)
        
        # Generate predictions
        pred_actions, pred_states = self.trajectory_generator(timesteps, states, actions, z_latent)
        pred_rewards = self.reward_predictor(z).squeeze(-1)
        
        return pred_actions, pred_states, pred_rewards

    def compute_loss(self, 
                    states: torch.Tensor, 
                    actions: torch.Tensor, 
                    timesteps: torch.Tensor, 
                    rewards: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute the ELBO loss:
        L(τ,y) = E_qφ[log p_β(τ|z) + log p_γ(y|z)] - KL(q_φ(z|τ,y) || p_α(z))
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            timesteps: Timestep indices [B, T]
            rewards: Reward sequence [B, T, 1]
            
        Returns:
            loss: Total loss (negative ELBO)
            loss_dict: Dictionary of individual loss components
        """
        # Encode trajectory
        mu, logvar = self.encode(states, actions, rewards)
        
        # Sample from latent distribution
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode and compute reconstruction losses
        pred_actions, pred_states, pred_rewards = self.decode(z, states, actions, timesteps)
        
        # Reconstruction losses (log p_β(τ|z) + log p_γ(y|z))
        action_loss = F.mse_loss(pred_actions, actions[:, -1, :])
        state_loss = F.mse_loss(pred_states, states[:, -1, :])
        reward_loss = F.mse_loss(pred_rewards, rewards[:, -1, 0])
        
        # KL divergence (KL(q_φ(z|τ,y) || p_α(z)))
        # Assuming p_α(z) is standard normal N(0,I)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total loss (negative ELBO)
        total_loss = (self.action_weight * (action_loss + state_loss) + 
                     self.reward_weight * reward_loss + 
                     kl_div)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_div': kl_div.item()
        }
        
        return total_loss, loss_dict

    def update_parameters(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor, 
                        timesteps: torch.Tensor, 
                        rewards: torch.Tensor) -> Dict[str, float]:
        """
        Update parameters using separate learning rates for each component:
        α_{t+1} = α_t + η₀ ∇_α L(τ,y)
        β_{t+1} = β_t + η₁ ∇_β L(τ,y)
        γ_{t+1} = γ_t + η₂ ∇_γ L(τ,y)
        φ_{t+1} = φ_t + η_φ ∇_φ L(τ,y)
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            timesteps: Timestep indices [B, T]
            rewards: Reward sequence [B, T, 1]
            
        Returns:
            Dictionary of loss values
        """
        # Zero gradients
        self.optimizer_alpha.zero_grad()
        self.optimizer_beta.zero_grad()
        self.optimizer_gamma.zero_grad()
        self.optimizer_phi.zero_grad()
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(states, actions, timesteps, rewards)
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters with respective learning rates
        self.optimizer_alpha.step()  # Update p_α(z)
        self.optimizer_beta.step()   # Update p_β(τ|z)
        self.optimizer_gamma.step()  # Update p_γ(y|z)
        self.optimizer_phi.step()    # Update q_φ(z|τ,y)
        
        return loss_dict

    def forward(self, 
                states: torch.Tensor, 
                actions: torch.Tensor, 
                timesteps: torch.Tensor, 
                rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference
        
        Args:
            states: State sequence [B, T, state_dim]
            actions: Action sequence [B, T, act_dim]
            timesteps: Timestep indices [B, T]
            rewards: Reward sequence [B, T, 1]
            
        Returns:
            pred_actions: Predicted actions [B, act_dim]
            pred_states: Predicted states [B, state_dim]
            pred_rewards: Predicted rewards [B]
        """
        # Encode trajectory
        mu, _ = self.encode(states, actions, rewards)
        
        # Use mean for inference
        z = mu.view(-1, self.n_latent, self.h_dim)
        
        # Decode and return predictions
        return self.decode(z, states, actions, timesteps) 