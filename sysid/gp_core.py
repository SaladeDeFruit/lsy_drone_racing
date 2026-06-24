"""A small, dependency-light Gaussian Process regressor (ARD-RBF kernel).

Implemented in numpy/scipy because scikit-learn is not available in this environment. It
fits an automatic-relevance-determination squared-exponential kernel by maximizing the log
marginal likelihood, and returns a predictive mean and standard deviation.

This is a single-output GP; the drone-specific wrapper (:mod:`sysid.residual_gp`) fits one
per output dimension.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

_JITTER = 1e-8


class GaussianProcess:
    """ARD squared-exponential GP regression with marginal-likelihood hyperparameters."""

    def __init__(self, n_restarts: int = 4, seed: int = 0):
        """Initialize the GP.

        Args:
            n_restarts: Number of random restarts for the hyperparameter optimization.
            seed: RNG seed for the restarts.
        """
        self._n_restarts = n_restarts
        self._rng = np.random.default_rng(seed)
        self._fitted = False

    # --- Kernel -------------------------------------------------------------------
    @staticmethod
    def _rbf(a: np.ndarray, b: np.ndarray, ls: np.ndarray, sf2: float) -> np.ndarray:
        """Signal-scaled ARD-RBF kernel between rows of ``a`` (m, d) and ``b`` (n, d)."""
        aw = a / ls
        bw = b / ls
        sq = (
            np.sum(aw**2, axis=1)[:, None]
            + np.sum(bw**2, axis=1)[None, :]
            - 2 * aw @ bw.T
        )
        return sf2 * np.exp(-0.5 * np.clip(sq, 0, None))

    def _unpack(self, theta: np.ndarray) -> tuple[float, np.ndarray, float]:
        """theta = [log_sf, log_ls(d), log_sn] -> (sf2, lengthscales, sn2)."""
        sf2 = np.exp(2 * theta[0])
        ls = np.exp(theta[1 : 1 + self._d])
        sn2 = np.exp(2 * theta[-1])
        return sf2, ls, sn2

    def _nlml(self, theta: np.ndarray) -> float:
        """Negative log marginal likelihood for hyperparameters ``theta``."""
        sf2, ls, sn2 = self._unpack(theta)
        K = self._rbf(self._X, self._X, ls, sf2) + (sn2 + _JITTER) * np.eye(self._n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((L, True), self._y)
        return float(
            0.5 * self._y @ alpha
            + np.sum(np.log(np.diag(L)))
            + 0.5 * self._n * np.log(2 * np.pi)
        )

    # --- Fit / predict ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Fit the GP to inputs ``X`` (n, d) and targets ``y`` (n,)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._n, self._d = X.shape

        # Standardize inputs, center targets (RBF assumes comparable feature scales).
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-9
        self._y_mean = y.mean()
        self._X = (X - self._x_mean) / self._x_std
        self._y = y - self._y_mean

        # Optimize hyperparameters from several random initializations.
        y_scale = np.log(np.std(self._y) + 1e-6)
        bounds = [(-6.0, 6.0)] + [(-4.0, 6.0)] * self._d + [(-12.0, 2.0)]
        best = None
        for r in range(self._n_restarts):
            if r == 0:
                theta0 = np.array([y_scale] + [0.0] * self._d + [-4.0])
            else:
                theta0 = self._rng.uniform(
                    [b[0] for b in bounds], [b[1] for b in bounds]
                )
            res = minimize(self._nlml, theta0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self._theta = best.x

        sf2, ls, sn2 = self._unpack(self._theta)
        self._sf2, self._ls, self._sn2 = sf2, ls, sn2
        K = self._rbf(self._X, self._X, ls, sf2) + (sn2 + _JITTER) * np.eye(self._n)
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, self._y)
        self._fitted = True
        return self

    def predict(self, Xs: np.ndarray, return_std: bool = True):
        """Predict mean (and optionally std) at test inputs ``Xs`` (m, d)."""
        if not self._fitted:
            raise RuntimeError("GaussianProcess.predict called before fit().")
        Xs = (np.asarray(Xs, dtype=float) - self._x_mean) / self._x_std
        Ks = self._rbf(Xs, self._X, self._ls, self._sf2)  # (m, n)
        mean = Ks @ self._alpha + self._y_mean
        if not return_std:
            return mean
        L = self._L[0]
        v = solve_triangular(L, Ks.T, lower=True)  # (n, m)
        var = self._sf2 - np.sum(v**2, axis=0)
        std = np.sqrt(np.clip(var, 0.0, None))
        return mean, std
