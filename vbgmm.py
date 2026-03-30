"""
Variational Bayesian Gaussian Mixture Model (VBGMM) — diagonal covariance.

Priors
------
  π  ~ Dirichlet(α_0)
  μ_kd ~ N(m_0d,  1/(β_0 · λ_kd))
  λ_kd ~ Gamma(a_0, b_0)          ← precision (1/variance) per dim

Variational family
------------------
  q(π)    = Dirichlet(α)
  q(μ_kd) = N(m_kd, 1/(β_k · ã_kd))   where ã_kd = a_kd / b_kd
  q(λ_kd) = Gamma(a_kd, b_kd)
  q(Z)    = ∏_n Categorical(r_n)

Key advantage over EM-GMM
--------------------------
  Redundant components are pruned automatically (α_k → α_0, N_k → 0).
  You can set K generously (e.g. 30) and let the model decide.
"""

import numpy as np
from scipy.special import digamma, gammaln


class VBGMM:
    def __init__(
        self,
        n_components: int = 20,
        alpha_0: float = 1.0 / 20,   # Dirichlet concentration (small → sparse)
        beta_0: float = 1.0,          # mean prior precision scaling
        a_0: float = 1e-3,            # Gamma shape prior
        b_0: float = 1e-3,            # Gamma rate prior
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.K = n_components
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.a_0 = a_0
        self.b_0 = b_0
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        # Variational parameters (set after fit)
        self.alpha_: np.ndarray = None   # (K,)   Dirichlet
        self.beta_:  np.ndarray = None   # (K,)   mean precision scaling
        self.m_:     np.ndarray = None   # (K, D) variational means
        self.a_:     np.ndarray = None   # (K, D) Gamma shape
        self.b_:     np.ndarray = None   # (K, D) Gamma rate
        self.elbos_: list = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _e_lnpi(self):
        """E[ln π_k] under Dirichlet q(π)."""
        return digamma(self.alpha_) - digamma(self.alpha_.sum())

    def _e_lnlambda(self):
        """E[ln λ_kd] = ψ(a_kd) - ln(b_kd).  Returns (K, D)."""
        return digamma(self.a_) - np.log(self.b_)

    def _e_lambda(self):
        """E[λ_kd] = a_kd / b_kd.  Returns (K, D)."""
        return self.a_ / self.b_

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_params(self, X: np.ndarray):
        N, D = X.shape
        # K-means++ for initial means
        idx = [self.rng.integers(N)]
        for _ in range(self.K - 1):
            dists = np.min(
                np.stack([np.sum((X - X[i]) ** 2, axis=1) for i in idx], axis=1),
                axis=1,
            )
            idx.append(self.rng.choice(N, p=dists / dists.sum()))
        self.m_ = X[idx].copy()                          # (K, D)

        self.alpha_ = np.full(self.K, self.alpha_0 + N / self.K)
        self.beta_  = np.full(self.K, self.beta_0 + N / self.K)
        var_global  = X.var(axis=0).mean()
        self.a_ = np.full((self.K, D), self.a_0 + 0.5)
        self.b_ = np.full((self.K, D), self.b_0 + 0.5 * var_global)

    # ------------------------------------------------------------------
    # E-step: update responsibilities
    # ------------------------------------------------------------------
    def _e_step(self, X: np.ndarray):
        N, D = X.shape
        E_lnpi  = self._e_lnpi()            # (K,)
        E_lnlam = self._e_lnlambda()        # (K, D)
        E_lam   = self._e_lambda()          # (K, D)

        # ln ρ_nk = E[ln π_k] + 0.5 Σ_d E[ln λ_kd]
        #           - 0.5 Σ_d E[λ_kd] * ((x_nd - m_kd)^2 + 1/β_k)
        ln_rho = np.zeros((N, self.K))
        for k in range(self.K):
            diff2 = (X - self.m_[k]) ** 2                   # (N, D)
            # mahalanobis-like term + uncertainty from q(μ)
            maha = (diff2 + 1.0 / self.beta_[k]) * E_lam[k]  # (N, D)
            ln_rho[:, k] = (
                E_lnpi[k]
                + 0.5 * E_lnlam[k].sum()
                - 0.5 * maha.sum(axis=1)
            )

        # log-normalise
        ln_norm = np.logaddexp.reduce(ln_rho, axis=1, keepdims=True)
        r = np.exp(ln_rho - ln_norm)         # (N, K)
        return r, ln_norm

    # ------------------------------------------------------------------
    # Full ELBO (monotone non-decreasing — used for convergence check)
    # ELBO = E_q[ln p(X,Z,π,μ,λ)] - E_q[ln q(Z,π,μ,λ)]
    # ------------------------------------------------------------------
    def _compute_elbo(self, X: np.ndarray, r: np.ndarray, ln_norm: np.ndarray) -> float:
        N, D = X.shape
        Nk = r.sum(axis=0) + 1e-10          # (K,)
        E_lam   = self._e_lambda()          # (K, D)
        E_lnlam = self._e_lnlambda()        # (K, D)
        E_lnpi  = self._e_lnpi()            # (K,)

        # 1. E[ln p(X | Z, μ, λ)]  — data log-likelihood
        elbo_data = ln_norm.sum()

        # 2. E[ln p(Z | π)] - E[ln q(Z)]  — entropy of assignments
        # = Σ_nk r_nk (E[ln π_k] - ln r_nk)
        # ln_norm already captures this; no separate term needed
        # (elbo_data = Σ_n log Σ_k ρ_nk already = E[ln p(X,Z|π,θ)] - E[ln q(Z)] )

        # 3. KL[ q(π) || p(π) ]
        alpha_sum = self.alpha_.sum()
        alpha_0_sum = self.alpha_0 * self.K
        kl_pi = (
            gammaln(alpha_0_sum) - self.K * gammaln(self.alpha_0)
            - gammaln(alpha_sum) + gammaln(self.alpha_).sum()
            + ((self.alpha_ - self.alpha_0) * E_lnpi).sum()
        )

        # 4. KL[ q(μ_k, λ_k) || p(μ_k, λ_k) ]  — summed over k, d
        kl_params = 0.0
        for k in range(self.K):
            # KL for λ_kd: Gamma(a_kd, b_kd) || Gamma(a_0, b_0)
            kl_lam = (
                (self.a_[k] - self.a_0) * digamma(self.a_[k])
                - gammaln(self.a_[k]) + gammaln(self.a_0)
                + self.a_0 * (np.log(self.b_[k]) - np.log(self.b_0))
                + self.a_[k] * (self.b_0 - self.b_[k]) / self.b_[k]
            ).sum()
            # KL for μ_kd | λ_kd: Normal(m_k, 1/(β_k λ_kd)) || Normal(0, 1/(β_0 λ_kd))
            kl_mu = 0.5 * (
                D * (self.beta_0 / self.beta_[k] - 1 + np.log(self.beta_[k] / self.beta_0))
                + self.beta_0 * E_lam[k].dot(self.m_[k] ** 2)
            )
            kl_params += kl_lam + kl_mu

        return float(elbo_data - kl_pi - kl_params)

    # ------------------------------------------------------------------
    # M-step: update variational parameters
    # ------------------------------------------------------------------
    def _m_step(self, X: np.ndarray, r: np.ndarray):
        N, D = X.shape
        Nk = r.sum(axis=0) + 1e-10          # (K,)

        # Dirichlet
        self.alpha_ = self.alpha_0 + Nk

        # Weighted statistics
        xbar = (r.T @ X) / Nk[:, None]      # (K, D)

        # Normal (mean)
        self.beta_ = self.beta_0 + Nk
        self.m_ = (self.beta_0 * 0.0 + Nk[:, None] * xbar) / self.beta_[:, None]
        # (prior mean m_0 = 0 for simplicity; data is PCA-transformed so ~0 mean)

        # Gamma (precision)
        for k in range(self.K):
            diff2 = (X - xbar[k]) ** 2       # (N, D)
            S_k = (r[:, k] @ diff2) / Nk[k]  # (D,)  weighted variance
            correction = (
                self.beta_0 * Nk[k] / self.beta_[k]
                * xbar[k] ** 2               # prior-data cross term
            )
            self.a_[k] = self.a_0 + (Nk[k] + 1.0) / 2.0
            self.b_[k] = self.b_0 + 0.5 * (Nk[k] * S_k + correction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "VBGMM":
        self._init_params(X)
        self.elbos_ = []
        prev_elbo = -np.inf

        for it in range(self.max_iter):
            r, ln_norm = self._e_step(X)
            self._m_step(X, r)
            elbo = self._compute_elbo(X, r, ln_norm)
            self.elbos_.append(elbo)

            delta = elbo - prev_elbo
            active = (self.alpha_ > self.alpha_0 + 0.5).sum()
            if self.verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:4d}  ELBO = {elbo:.2f}  "
                      f"delta = {delta:.4f}  active_k = {active}")
            if delta < self.tol and it > 0:
                if self.verbose:
                    print(f"  Converged at iter {it}  active_k = {active}")
                break
            prev_elbo = elbo

        self._print_active_components()
        return self

    def _print_active_components(self):
        weights = self.alpha_ / self.alpha_.sum()
        active = np.where(weights > 1e-3)[0]
        print(f"\n  Active components ({len(active)}/{self.K}):")
        for k in active:
            print(f"    k={k:2d}  weight={weights[k]:.4f}")

    def sample(self, n_samples: int) -> np.ndarray:
        D = self.m_.shape[1]
        weights = self.alpha_ / self.alpha_.sum()
        k_ids = self.rng.choice(self.K, size=n_samples, p=weights)
        samples = np.zeros((n_samples, D))
        E_lam = self._e_lambda()             # (K, D)
        for k in range(self.K):
            mask = k_ids == k
            cnt = mask.sum()
            if cnt == 0:
                continue
            std = np.sqrt(1.0 / E_lam[k])   # posterior mean std per dim
            samples[mask] = self.m_[k] + self.rng.standard_normal((cnt, D)) * std
        return samples

    def score(self, X: np.ndarray) -> float:
        r, ln_norm = self._e_step(X)
        return self._compute_elbo(X, r, ln_norm) / len(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        r, _ = self._e_step(X)
        return r.argmax(axis=1)

    @property
    def effective_n_components(self) -> int:
        weights = self.alpha_ / self.alpha_.sum()
        return int((weights > 1e-3).sum())
