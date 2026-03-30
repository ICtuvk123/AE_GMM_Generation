"""
Training script: AE + GMM / VBGMM on MNIST.

Two modes:
  --mode perclass  : one model per digit class (default)
  --mode global    : single model over all digits

Two model types:
  --model gmm    : EM-GMM (fixed K)
  --model vbgmm  : Variational Bayes GMM (auto-prunes K)
"""
import argparse
import os
import pickle
import numpy as np

from data_loader import load_mnist
from gmm import GMM
from representation import fit_representation
from vbgmm import VBGMM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data")
    p.add_argument("--save_dir",     default="checkpoints")
    
    # AE parameters
    p.add_argument("--ae_latent_dim", type=int, default=32,
                   help="AE latent dimension")
    p.add_argument("--ae_epochs",    type=int, default=30,
                   help="AE training epochs")
    p.add_argument("--ae_batch_size", type=int, default=256,
                   help="AE training batch size")
    p.add_argument("--ae_lr",        type=float, default=1e-3,
                   help="AE learning rate")
    p.add_argument("--ae_ckpt",      default=None,
                   help="Optional pretrained AE checkpoint to reuse")
    
    # GMM/VBGMM parameters
    p.add_argument("--n_components", type=int, default=20,
                   help="Max GMM components (VBGMM will auto-prune)")
    p.add_argument("--cov_type",     default="diag",
                   choices=["full", "diag", "spherical"],
                   help="Covariance type (GMM only)")
    p.add_argument("--max_iter",     type=int, default=200)
    
    # Training mode
    p.add_argument("--mode",         default="perclass",
                   choices=["perclass", "global"])
    p.add_argument("--model",        default="vbgmm",
                   choices=["gmm", "vbgmm"],
                   help="Model type: gmm (EM) or vbgmm (Variational Bayes)")
    p.add_argument("--alpha_0",      type=float, default=None,
                   help="Dirichlet prior concentration for VBGMM "
                        "(default: 1/K, smaller → sparser)")
    return p.parse_args()


def _build_model(args):
    if args.model == "gmm":
        return GMM(
            n_components=args.n_components,
            covariance_type=args.cov_type,
            max_iter=args.max_iter,
        )
    else:
        alpha_0 = args.alpha_0 if args.alpha_0 else 1.0 / args.n_components
        return VBGMM(
            n_components=args.n_components,
            alpha_0=alpha_0,
            max_iter=args.max_iter,
        )


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=== Loading MNIST ===")
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"\n=== Training Autoencoder ===")
    repr_info, Z_train, Z_test = fit_representation(
        X_train,
        X_test,
        save_dir=args.save_dir,
        ae_latent_dim=args.ae_latent_dim,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        ae_ckpt=args.ae_ckpt,
    )
    print(repr_info["repr_summary"])

    if args.mode == "perclass":
        _train_perclass(args, repr_info, Z_train, y_train, Z_test, y_test)
    else:
        _train_global(args, repr_info, Z_train, Z_test)


def _train_perclass(args, repr_info, Z_train, y_train, Z_test, y_test):
    print(f"\n=== Per-class {args.model.upper()} (K={args.n_components}) ===")
    models = {}
    for digit in range(10):
        print(f"\n-- Digit {digit} --")
        mask = y_train == digit
        Z_d = Z_train[mask]
        model = _build_model(args)
        model.fit(Z_d)
        test_score = model.score(Z_test[y_test == digit])
        print(f"  Test score/sample: {test_score:.4f}")
        if args.model == "vbgmm":
            print(f"  Effective components: {model.effective_n_components}/{args.n_components}")
        models[digit] = model

    fname = f"model_perclass_ae_{args.model}_k{args.n_components}.pkl"
    ckpt = {
        "mode": "perclass",
        "model_type": args.model,
        **repr_info,
        "models": models,
        "args": vars(args),
    }
    path = os.path.join(args.save_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"\nSaved to {path}")


def _train_global(args, repr_info, Z_train, Z_test):
    print(f"\n=== Global {args.model.upper()} (K={args.n_components}) ===")
    model = _build_model(args)
    model.fit(Z_train)
    test_score = model.score(Z_test)
    print(f"Test score/sample: {test_score:.4f}")
    if args.model == "vbgmm":
        print(f"Effective components: {model.effective_n_components}/{args.n_components}")

    fname = f"model_global_ae_{args.model}_k{args.n_components}.pkl"
    ckpt = {
        "mode": "global",
        "model_type": args.model,
        **repr_info,
        "model": model,
        "args": vars(args),
    }
    path = os.path.join(args.save_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
