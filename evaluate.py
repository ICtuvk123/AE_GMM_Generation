"""
Evaluation utilities:
  1. Test-set score (log-likelihood for GMM, ELBO proxy for VBGMM)
  2. GMM component count vs log-likelihood (model selection)
  3. t-SNE visualisation of generated vs real samples
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_mnist
from representation import describe_representation, encode_with_ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--out_dir",   default="outputs/eval")
    p.add_argument("--ckpt",      default=None,
                   help="Path to saved checkpoint for test-set evaluation")
    p.add_argument("--k_sweep",   action="store_true",
                   help="Sweep GMM n_components and plot log-likelihood")
    p.add_argument("--tsne",      action="store_true",
                   help="t-SNE plot of generated vs real samples")
    p.add_argument("--digit",     type=int, default=3,
                   help="Digit to use for k_sweep / tsne")
    return p.parse_args()


# ------------------------------------------------------------------
def k_sweep(X_digit, out_dir, digit=0,
            k_list=None, max_iter=100):
    """Sweep over number of GMM components and plot validation log-likelihood."""
    if k_list is None:
        k_list = [1, 2, 4, 8, 12, 16, 24, 32]

    # 80/20 split
    split = int(0.8 * len(X_digit))
    Z_tr, Z_val = X_digit[:split], X_digit[split:]

    scores = []
    for k in k_list:
        print(f"  K={k} ...", end=" ", flush=True)
        from gmm import GMM
        gmm = GMM(n_components=k, covariance_type="diag",
                  max_iter=max_iter, verbose=False)
        gmm.fit(Z_tr)
        s = gmm.score(Z_val)
        scores.append(s)
        print(f"val ll/sample = {s:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(k_list, scores, marker="o", ms=5)
    plt.xlabel("Number of GMM components K")
    plt.ylabel("Validation log-likelihood / sample")
    plt.title(f"GMM model selection  (digit={digit})")
    plt.grid(True)
    path = os.path.join(out_dir, f"k_sweep_digit{digit}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------
def tsne_plot(X_real, ckpt, model, out_dir, ckpt_path, digit=0, n=300):
    """t-SNE visualization of real vs generated samples in latent space."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not available, skipping t-SNE")
        return

    Z_real = encode_with_ckpt(ckpt, X_real[:n], base_ckpt_path=ckpt_path)
    Z_gen  = model.sample(n)
    Z_all  = np.vstack([Z_real, Z_gen])

    print("  Running t-SNE ...")
    emb = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Z_all)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=0.5, label="real")
    plt.scatter(emb[n:, 0], emb[n:, 1], s=8, alpha=0.5, label="generated")
    plt.legend(); plt.axis("off")
    plt.title(f"t-SNE: real vs generated  (digit={digit})")
    path = os.path.join(out_dir, f"tsne_digit{digit}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading MNIST ...")
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)

    if args.k_sweep:
        print(f"\n=== K sweep for digit {args.digit} ===")
        X_digit = X_train[y_train == args.digit]
        k_sweep(X_digit, args.out_dir, digit=args.digit)

    if args.ckpt:
        print(f"\n=== Test-set evaluation from {args.ckpt} ===")
        with open(args.ckpt, "rb") as f:
            ckpt = pickle.load(f)
        mode = ckpt["mode"]
        print(f"  Representation: {describe_representation(ckpt)}")
        if mode == "perclass":
            for digit in range(10):
                model = ckpt["models"][digit]
                Z = encode_with_ckpt(ckpt, X_test[y_test == digit], base_ckpt_path=args.ckpt)
                print(f"  Digit {digit}  test score/sample = {model.score(Z):.4f}")
                if args.tsne and digit == args.digit:
                    tsne_plot(
                        X_test[y_test == digit],
                        ckpt,
                        model,
                        args.out_dir,
                        args.ckpt,
                        digit=digit,
                    )
        else:
            model = ckpt["model"]
            Z = encode_with_ckpt(ckpt, X_test, base_ckpt_path=args.ckpt)
            print(f"  Global test score/sample = {model.score(Z):.4f}")
            if args.tsne:
                tsne_plot(X_test, ckpt, model, args.out_dir, args.ckpt, digit="all")


if __name__ == "__main__":
    main()
