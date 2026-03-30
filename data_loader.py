"""MNIST data loader.

Priority order:
  1. torchvision (most reliable, handles download automatically)
  2. Direct download from working mirrors
  3. Load from already-downloaded raw files
"""
import numpy as np
import os
import struct
import gzip


# Mirrors in priority order (yann.lecun.com is defunct)
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _try_torchvision(data_dir):
    """Use torchvision to download MNIST. Returns (X_train, y_train, X_test, y_test) or None."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        print("Using torchvision to download MNIST ...")
        transform = transforms.ToTensor()
        train_ds = torchvision.datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_train = train_ds.targets.numpy().astype(np.int32)
        X_test  = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test  = test_ds.targets.numpy().astype(np.int32)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"torchvision not available or failed: {e}")
        return None


def _download_raw(data_dir):
    """Download raw .gz files from mirrors."""
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for name, fname in FILES.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            continue
        downloaded = False
        for mirror in MIRRORS:
            url = mirror + fname
            try:
                print(f"Downloading {fname} from {mirror} ...")
                urllib.request.urlretrieve(url, fpath)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed ({e}), trying next mirror ...")
        if not downloaded:
            raise RuntimeError(
                f"Could not download {fname} from any mirror.\n"
                "Please download MNIST manually and place the .gz files in: "
                + os.path.abspath(data_dir)
            )


def _load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h * w).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def _load_from_raw(data_dir):
    X_train = _load_images(os.path.join(data_dir, FILES["train_images"]))
    y_train = _load_labels(os.path.join(data_dir, FILES["train_labels"]))
    X_test  = _load_images(os.path.join(data_dir, FILES["test_images"]))
    y_test  = _load_labels(os.path.join(data_dir, FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


def load_mnist(data_dir="data", download=True):
    """Return (X_train, y_train, X_test, y_test), images in [0,1] float32."""
    if not download:
        return _load_from_raw(data_dir)

    # Strategy 1: torchvision
    result = _try_torchvision(data_dir)
    if result is not None:
        return result

    # Strategy 2: raw file download + parse
    _download_raw(data_dir)
    return _load_from_raw(data_dir)
