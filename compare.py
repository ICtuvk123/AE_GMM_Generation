"""Save real MNIST samples for visual comparison with generated images."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import load_mnist

X, y, _, _ = load_mnist()
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(X[idx].reshape(28, 28), cmap="gray")
    axes[0, digit].set_title(str(digit))
    axes[0, digit].axis("off")
    idx2 = np.where(y == digit)[0][1]
    axes[1, digit].imshow(X[idx2].reshape(28, 28), cmap="gray")
    axes[1, digit].axis("off")
plt.suptitle("Real MNIST samples")
plt.savefig("outputs/real_samples.png", dpi=150, bbox_inches="tight")
print("Saved outputs/real_samples.png")
