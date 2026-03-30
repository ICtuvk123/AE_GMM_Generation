"""
Autoencoder for MNIST.

Encoder: 784 -> 512 -> 256 -> latent_dim
Decoder: latent_dim -> 256 -> 512 -> 784
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, 784),        nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """numpy (N, 784) -> numpy (N, latent_dim)"""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return self.encoder(t).cpu().numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        """numpy (N, latent_dim) -> numpy (N, 784) in [0,1]"""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32)
            return self.decoder(t).cpu().numpy()


def train_ae(
    X_train: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: str = "checkpoints/autoencoder.pt",
    device: str = None,
) -> Autoencoder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training AE on {device}  (latent_dim={latent_dim}, epochs={epochs})")

    model = Autoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            recon = model(x)
            loss  = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        avg = total_loss / len(X_train)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss = {avg:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "latent_dim": latent_dim}, save_path)
    print(f"AE saved to {save_path}")
    return model


def load_ae(path: str) -> Autoencoder:
    ckpt = torch.load(path, map_location="cpu")
    model = Autoencoder(ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
