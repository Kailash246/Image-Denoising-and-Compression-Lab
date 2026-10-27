"""Convolutional VAE for 32x32 RGB patches."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 32


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),     # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),    # 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 4x4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 128, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(recon, target, mu, logvar, beta: float = 0.01):
    """Per-image summed MSE + beta * KL, averaged over the batch."""
    b = target.size(0)
    rec = F.mse_loss(recon, target, reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return rec + beta * kl, rec, kl
