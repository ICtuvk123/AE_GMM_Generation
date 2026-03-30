"""
Generation script: sample from trained GMM and visualise results.

Usage examples:
  python generate.py --ckpt checkpoints/model_perclass_ae_gmm_k10.pkl --digit 3 --n 64
  python generate.py --ckpt checkpoints/model_global_ae_vbgmm.pkl    --n 100
  python generate.py --ckpt checkpoints/model_perclass_ae_vbgmm.pkl  --show_prototypes
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representation import decode_with_ckpt, describe_representation, get_component_means


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  default="checkpoints/model_perclass.pkl")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--tag", default=None,
                   help="Optional filename prefix for saved images. "
                        "Defaults to checkpoint stem.")
    p.add_argument("--digit", type=int, default=None,
                   help="Which digit to generate (perclass mode only). "
                        "If None, generate all digits.")
    p.add_argument("--n",     type=int, default=64,  help="Samples per digit")
    p.add_argument("--show_prototypes", action="store_true",
                   help="Visualise GMM component means as prototype images")
    p.add_argument("--sharpen", action="store_true",
                   help="Apply contrast stretch + gamma to reduce blurriness")
    p.add_argument("--gamma", type=float, default=1.5,
                   help="Gamma value for sharpening (>1 darker/crisper, default 1.5)")
    return p.parse_args()


def _sharpen(images, gamma=1.5):
    """Per-image contrast stretch + gamma, makes edges crisper."""
    out = np.empty_like(images)
    for i, img in enumerate(images):
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)   # stretch to [0,1]
        out[i] = np.power(img, gamma)       # gamma < 1 brightens, > 1 darkens
    return out


def _to_image(z, ckpt, ckpt_path, sharpen=False, gamma=1.5):
    """z: (D_repr,) or (N, D_repr) -> image array in [0,1]."""
    x = decode_with_ckpt(ckpt, np.atleast_2d(z), base_ckpt_path=ckpt_path)
    imgs = np.clip(x, 0, 1).reshape(-1, 28, 28)
    if sharpen:
        imgs = _sharpen(imgs, gamma=gamma)
    return imgs


def _grid(images, ncols=8, pad=2):
    """images: (N, 28, 28) -> single grid image."""
    N = len(images)
    nrows = (N + ncols - 1) // ncols
    h, w = 28 + pad, 28 + pad
    grid = np.ones((nrows * h + pad, ncols * w + pad))
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r*h+pad:(r+1)*h, c*w+pad:(c+1)*w] = img
    return grid


def save_grid(images, path, title="", ncols=8):
    grid = _grid(images, ncols=ncols)
    fig, ax = plt.subplots(figsize=(ncols * 1.2, (len(images)//ncols + 1) * 1.2))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _prefixed_name(args, name):
    prefix = args.tag if args.tag else os.path.splitext(os.path.basename(args.ckpt))[0]
    return os.path.join(args.out_dir, f"{prefix}_{name}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)

    mode = ckpt["mode"]
    print(f"Loaded checkpoint  mode={mode}  repr={describe_representation(ckpt)}")

    if mode == "perclass":
        models = ckpt["models"]
        digits = [args.digit] if args.digit is not None else list(range(10))

        all_imgs = []
        for digit in digits:
            model = models[digit]
            Z = model.sample(args.n)
            imgs = _to_image(Z, ckpt, args.ckpt, sharpen=args.sharpen, gamma=args.gamma)
            all_imgs.append(imgs)
            out = _prefixed_name(args, f"digit_{digit}.png")
            save_grid(imgs, out, title=f"Digit {digit}  ({args.n} samples)", ncols=8)

            if args.show_prototypes:
                protos = _to_image(
                    get_component_means(model),
                    ckpt,
                    args.ckpt,
                    sharpen=args.sharpen,
                    gamma=args.gamma,
                )
                out_p = _prefixed_name(args, f"proto_{digit}.png")
                save_grid(protos, out_p, title=f"Prototypes digit {digit}", ncols=min(10, model.K))

        # Combined grid: one row per digit
        if len(digits) > 1:
            combined = np.concatenate(all_imgs, axis=0)
            save_grid(combined, _prefixed_name(args, "all_digits.png"),
                      title="All digits", ncols=args.n)

    else:  # global
        model = ckpt["model"]
        Z = model.sample(args.n)
        imgs = _to_image(Z, ckpt, args.ckpt, sharpen=args.sharpen, gamma=args.gamma)
        save_grid(imgs, _prefixed_name(args, "global_samples.png"),
                  title=f"Global model  ({args.n} samples)")

        if args.show_prototypes:
            protos = _to_image(
                get_component_means(model),
                ckpt,
                args.ckpt,
                sharpen=args.sharpen,
                gamma=args.gamma,
            )
            save_grid(protos, _prefixed_name(args, "global_prototypes.png"),
                      title="Mixture component means", ncols=min(10, model.K))


if __name__ == "__main__":
    main()
