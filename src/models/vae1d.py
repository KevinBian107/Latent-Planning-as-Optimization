import torch
import torch.nn as nn
import torch

class VAE1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # We interpret `dim` as the flattened latent dimension (z_dim)
        self.z_dim = dim

        # Encoder: maps input z -> hidden -> latent parameters
        self.encoder = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )
        # Latent mean and log-variance
        self.fc_mu = nn.Linear(self.z_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.z_dim, self.z_dim)

        # Decoder: maps sampled latent back to reconstructed z
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            # no activation or use identity
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, channels=1, z_dim)
        returns: reconstructed Tensor of same shape
        """
        B, C, L = x.shape  # C should be 1, L == z_dim
        # flatten across channel
        h = x.view(B, L)

        # encode to latent distribution
        h_enc = self.encoder(h)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # decode
        recon = self.decoder(z)
        recon = recon.view(B, C, L)
        return recon
