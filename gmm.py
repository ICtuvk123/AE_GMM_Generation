import numpy as np


class GMM:
    """
    Gaussian Mixture Model with EM algorithm.
    
    Parameters
    ----------
    n_components : int
        Number of mixture components.
    covariance_type : str
        Type of covariance: "full", "diag", or "spherical".
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_components=10, covariance_type="full", max_iter=200, tol=1e-5, seed=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        # Fitted parameters (set after fit())
        self.pi_ = None  # mixing coefficients (K,)
        self.mu_ = None  # means (K, D)
        self.cov_ = None  # covariances (K, D, D) or (K, D) for diag
        self.n_features_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.ll_hist_ = []

    @property
    def K(self):
        """Number of components (alias for n_components)."""
        return self.n_components

    @property
    def effective_n_components(self):
        """Number of active components (components with non-negligible weight)."""
        if self.pi_ is None:
            return self.n_components
        return int((self.pi_ > 1e-3).sum())

    def fit(self, X):
        """
        Fit the GMM to data X.
        
        Parameters
        ----------
        X : ndarray (N, D)
            Training data.
        
        Returns
        -------
        self
        """
        result = fit_gmm_em(
            X,
            K=self.n_components,
            max_iters=self.max_iter,
            tol=self.tol,
            seed=self.seed,
            cov_type=self.covariance_type,
        )
        self.pi_ = result["pi"]
        self.mu_ = result["mu"]
        self.cov_ = result["cov"]
        self.n_features_ = X.shape[1]
        self.converged_ = result["converged"]
        self.n_iter_ = result["iters"]
        self.ll_hist_ = result["ll_hist"]
        return self

    def _compute_log_likelihood(self, X):
        """Compute log-likelihood of X under fitted model."""
        if self.pi_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return mean_log_likelihood(X, self.pi_, self.mu_, self.cov_)

    def score(self, X):
        """
        Compute average log-likelihood per sample.
        
        Parameters
        ----------
        X : ndarray (N, D)
            Data to evaluate.
        
        Returns
        -------
        float
            Average log-likelihood per sample.
        """
        ll = self._compute_log_likelihood(X)
        return ll

    def sample(self, n_samples):
        """
        Generate samples from the fitted GMM.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        
        Returns
        -------
        ndarray (n_samples, D)
            Generated samples.
        """
        if self.pi_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return sample_from_gmm(self.pi_, self.mu_, self.cov_, n_samples, seed=self.seed)

    def predict(self, X):
        """
        Predict cluster assignments for X.
        
        Parameters
        ----------
        X : ndarray (N, D)
            Data points.
        
        Returns
        -------
        ndarray (N,)
            Cluster indices.
        """
        if self.pi_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return predict(X, self.pi_, self.mu_, self.cov_)

    def predict_proba(self, X):
        """
        Predict posterior probabilities for each component.
        
        Parameters
        ----------
        X : ndarray (N, D)
            Data points.
        
        Returns
        -------
        ndarray (N, K)
            Posterior probabilities.
        """
        if self.pi_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return e_step(X, self.pi_, self.mu_, self.cov_)


def init_params(X, K=2, seed=None, cov_type="full"):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    
    pi = np.full(K, 1.0 / K)
    idx = [rng.integers(N)]
    for _ in range(K - 1):
        dists = np.min(
            np.stack([np.sum((X - X[i]) ** 2, axis=1) for i in idx], axis=1),
            axis=1,
        )
        probs = dists / dists.sum()
        idx.append(rng.choice(N, p=probs))
    mu = X[idx].copy()
    
    var = X.var(axis=0).mean()
    cov = np.stack([np.eye(D) * var for _ in range(K)])
    
    return pi, mu, cov


def log_mvn(X, mu, cov):
    D = X.shape[1]
    cov_reg = (cov + cov.T) / 2 + 1e-6 * np.eye(D)
    L = np.linalg.cholesky(cov_reg)
    diff = X - mu
    sol = np.linalg.solve(L, diff.T)
    quad = np.sum(sol ** 2, axis=0)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def e_step(X, pi, mu, cov):
    K = len(pi)
    logs = [np.log(pi[k] + 1e-12) + log_mvn(X, mu[k], cov[k]) for k in range(K)]
    log_comp = np.stack(logs, axis=1)
    m = log_comp.max(axis=1, keepdims=True)
    log_gamma = log_comp - (m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True)))
    return np.exp(log_gamma)


def m_step(X, gamma):
    N, D = X.shape
    K = gamma.shape[1]
    Nk = gamma.sum(axis=0) + 1e-12
    
    pi = Nk / N
    mu = (gamma.T @ X) / Nk[:, None]
    
    cov = np.empty((K, D, D))
    for k in range(K):
        Xc = X - mu[k]
        cov[k] = (Xc.T * gamma[:, k]) @ Xc / Nk[k]
        cov[k] = (cov[k] + cov[k].T) / 2 + 1e-6 * np.eye(D)
    
    return pi, mu, cov


def mean_log_likelihood(X, pi, mu, cov):
    K = len(pi)
    log_comp = np.stack([np.log(pi[k] + 1e-12) + log_mvn(X, mu[k], cov[k]) for k in range(K)], axis=1)
    m = log_comp.max(axis=1, keepdims=True)
    log_sum = m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True))
    return float(log_sum.mean())


def fit_gmm_em(X, K=2, max_iters=200, tol=1e-5, seed=None, cov_type="full"):
    pi, mu, cov = init_params(X, K=K, seed=seed, cov_type=cov_type)
    ll_hist = []
    
    for it in range(max_iters):
        gamma = e_step(X, pi, mu, cov)
        pi, mu, cov = m_step(X, gamma)
        ll = mean_log_likelihood(X, pi, mu, cov)
        ll_hist.append(ll)
        
        if it > 0 and (ll_hist[-1] - ll_hist[-2]) < tol:
            break
    
    return {
        'pi': pi,
        'mu': mu,
        'cov': cov,
        'gamma': gamma,
        'iters': it + 1,
        'll_hist': ll_hist,
        'converged': it + 1 < max_iters,
    }


def sample_from_gmm(pi, mu, cov, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    K, D = mu.shape
    
    k_ids = rng.choice(K, size=n_samples, p=pi)
    samples = np.zeros((n_samples, D))
    
    for k in range(K):
        mask = k_ids == k
        cnt = mask.sum()
        if cnt == 0:
            continue
        samples[mask] = rng.multivariate_normal(mu[k], cov[k], cnt)
    
    return samples


def predict(X, pi, mu, cov):
    gamma = e_step(X, pi, mu, cov)
    return gamma.argmax(axis=1)
