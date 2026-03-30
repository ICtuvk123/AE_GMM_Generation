"""Autoencoder representation backend for latent-space mixture modelling."""
import os
from typing import Any, Dict, Tuple

import numpy as np

from autoencoder import load_ae, train_ae


def fit_representation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    save_dir: str = "checkpoints",
    ae_latent_dim: int = 32,
    ae_epochs: int = 30,
    ae_batch_size: int = 256,
    ae_lr: float = 1e-3,
    ae_ckpt: str | None = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Fit autoencoder and return latent train/test features."""
    os.makedirs(save_dir, exist_ok=True)
    
    if ae_ckpt:
        ae = load_ae(ae_ckpt)
        ae_path = ae_ckpt
        ae_ref = os.path.abspath(ae_ckpt)
        latent_dim = ae.latent_dim
        print(f"Loaded AE from {ae_ckpt}")
    else:
        ae_path = os.path.join(save_dir, f"autoencoder_ld{ae_latent_dim}.pt")
        ae_ref = os.path.abspath(ae_path)
        ae = train_ae(
            X_train,
            latent_dim=ae_latent_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            lr=ae_lr,
            save_path=ae_path,
        )
        latent_dim = ae_latent_dim

    Z_train = ae.encode(X_train)
    Z_test = ae.encode(X_test)
    info = {
        "repr_type": "ae",
        "ae_ckpt": ae_ref,
        "repr_dims": latent_dim,
        "repr_summary": f"AE(latent_dim={latent_dim})",
    }
    return info, Z_train, Z_test


def describe_representation(ckpt: Dict[str, Any]) -> str:
    """Return a human-readable description of the representation."""
    return f"AE(latent_dim={ckpt.get('repr_dims', '?')})"


def _resolve_repr_path(stored_path: str, base_ckpt_path: str | None = None) -> str:
    if os.path.isabs(stored_path):
        return stored_path
    if base_ckpt_path is None:
        return stored_path
    return os.path.join(os.path.dirname(base_ckpt_path), stored_path)


def encode_with_ckpt(ckpt: Dict[str, Any], X: np.ndarray, *, base_ckpt_path: str | None = None) -> np.ndarray:
    """Encode images using the AE from checkpoint."""
    ae_path = _resolve_repr_path(ckpt["ae_ckpt"], base_ckpt_path)
    ae = load_ae(ae_path)
    return ae.encode(X)


def decode_with_ckpt(ckpt: Dict[str, Any], Z: np.ndarray, *, base_ckpt_path: str | None = None) -> np.ndarray:
    """Decode latent vectors using the AE from checkpoint."""
    ae_path = _resolve_repr_path(ckpt["ae_ckpt"], base_ckpt_path)
    ae = load_ae(ae_path)
    return ae.decode(Z)


def get_component_means(model) -> np.ndarray:
    """Return component means for either GMM or VBGMM."""
    if hasattr(model, "mu_") and model.mu_ is not None:
        return model.mu_
    if hasattr(model, "m_") and model.m_ is not None:
        return model.m_
    raise AttributeError("Model does not expose component means")
