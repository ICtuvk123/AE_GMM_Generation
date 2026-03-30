"""
Gaussian Mixture Model trained with the EM algorithm.

Supports three covariance types:
  - 'full'      : each component has its own full covariance matrix
  - 'diag'      : each component has a diagonal covariance matrix
  - 'spherical' : each component has a single variance scalar
"""
import numpy as np


class GMM:
    def __init__(
        self,
        n_components: int = 10,
        covariance_type: str = "diag",
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42,
        verbose: bool = True,
    ):
        assert covariance_type in ("full", "diag", "spherical")
        self.K = n_components
        self.cov_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg_covar
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        # Parameters (set after fit)
        self.pi_: np.ndarray = None    # (K,)  mixing weights
        self.mu_: np.ndarray = None    # (K, D) means
        self.cov_: np.ndarray = None   # shape depends on covariance_type
        self.log_likelihoods_: list = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_params(self, X: np.ndarray):
        N, D = X.shape
        self.pi_ = np.full(self.K, 1.0 / self.K)
        # K-means++ style initialisation for means
        idx = [self.rng.integers(N)]
        for _ in range(self.K - 1):
            dists = np.min(
                np.stack([np.sum((X - X[i]) ** 2, axis=1) for i in idx], axis=1),
                axis=1,
            )
            probs = dists / dists.sum()
            idx.append(self.rng.choice(N, p=probs))
        self.mu_ = X[idx].copy()

        # Covariance initialisation: use global variance
        var = X.var(axis=0).mean()
        if self.cov_type == "full":
            self.cov_ = np.stack([np.eye(D) * var for _ in range(self.K)])   # (K,D,D)
        elif self.cov_type == "diag":
            self.cov_ = np.full((self.K, D), var)                             # (K,D)
        else:  # spherical
            self.cov_ = np.full(self.K, var)                                  # (K,)

    # ------------------------------------------------------------------
    # Log-likelihood helpers
    # ------------------------------------------------------------------
    def _log_prob(self, X: np.ndarray) -> np.ndarray:
        """Return (N, K) log N(x | mu_k, Sigma_k)."""
        N, D = X.shape
        log_p = np.zeros((N, self.K))
        for k in range(self.K):
            diff = X - self.mu_[k]                   # (N, D)
            if self.cov_type == "full":
                Sigma = self.cov_[k] + np.eye(D) * self.reg
                sign, logdet = np.linalg.slogdet(Sigma)
                L = np.linalg.cholesky(Sigma)
                maha = np.sum(np.linalg.solve(L, diff.T) ** 2, axis=0)  # (N,)
            elif self.cov_type == "diag":
                var = self.cov_[k] + self.reg          # (D,)
                logdet = np.sum(np.log(var))
                maha = np.sum(diff ** 2 / var, axis=1) # (N,)
            else:  # spherical
                var = self.cov_[k] + self.reg
                logdet = D * np.log(var)
                maha = np.sum(diff ** 2, axis=1) / var
            log_p[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + maha)
        return log_p

    def _e_step(self, X: np.ndarray):
        """Return responsibilities (N, K) and log-likelihood scalar."""
        log_p = self._log_prob(X)                    # (N, K)
        log_weighted = log_p + np.log(self.pi_)      # (N, K)
        # log-sum-exp for numerical stability
        log_norm = np.logaddexp.reduce(log_weighted, axis=1, keepdims=True)  # (N,1)
        log_r = log_weighted - log_norm              # (N, K)
        r = np.exp(log_r)                            # (N, K)
        log_likelihood = log_norm.sum()
        return r, log_likelihood

    def _m_step(self, X: np.ndarray, r: np.ndarray):
        N, D = X.shape
        Nk = r.sum(axis=0) + 1e-10                  # (K,)
        self.pi_ = Nk / N
        self.mu_ = (r.T @ X) / Nk[:, None]          # (K, D)

        if self.cov_type == "full":
            cov = np.zeros((self.K, D, D))
            for k in range(self.K):
                diff = X - self.mu_[k]               # (N, D)
                cov[k] = (r[:, k:k+1] * diff).T @ diff / Nk[k]
                cov[k] += np.eye(D) * self.reg
            self.cov_ = cov
        elif self.cov_type == "diag":
            cov = np.zeros((self.K, D))
            for k in range(self.K):
                diff = X - self.mu_[k]
                cov[k] = (r[:, k] @ (diff ** 2)) / Nk[k] + self.reg
            self.cov_ = cov
        else:  # spherical
            cov = np.zeros(self.K)
            for k in range(self.K):
                diff = X - self.mu_[k]
                cov[k] = (r[:, k] @ np.sum(diff ** 2, axis=1)) / (Nk[k] * D) + self.reg
            self.cov_ = cov

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GMM":
        self._init_params(X)
        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for it in range(self.max_iter):
            r, ll = self._e_step(X)
            self._m_step(X, r)
            self.log_likelihoods_.append(ll)

            delta = ll - prev_ll
            if self.verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:4d}  log-likelihood = {ll:.4f}  delta = {delta:.4f}")
            if abs(delta) < self.tol and it > 0:
                if self.verbose:
                    print(f"  Converged at iter {it} (delta={delta:.2e})")
                break
            prev_ll = ll
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from the fitted GMM. Returns (n_samples, D)."""
        D = self.mu_.shape[1]
        # Choose components
        k_ids = self.rng.choice(self.K, size=n_samples, p=self.pi_)
        samples = np.zeros((n_samples, D))
        for k in range(self.K):
            mask = k_ids == k
            cnt = mask.sum()
            if cnt == 0:
                continue
            if self.cov_type == "full":
                samples[mask] = self.rng.multivariate_normal(self.mu_[k], self.cov_[k], cnt)
            elif self.cov_type == "diag":
                samples[mask] = self.mu_[k] + self.rng.standard_normal((cnt, D)) * np.sqrt(self.cov_[k])
            else:
                samples[mask] = self.mu_[k] + self.rng.standard_normal((cnt, D)) * np.sqrt(self.cov_[k])
        return samples

    def score(self, X: np.ndarray) -> float:
        """Average log-likelihood per sample."""
        _, ll = self._e_step(X)
        return ll / len(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard assignment to most likely component."""
        r, _ = self._e_step(X)
        return r.argmax(axis=1)
