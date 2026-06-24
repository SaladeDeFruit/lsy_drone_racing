"""Gaussian-Process residual model for offline drone-model identification.

Learns the discrepancy between the nominal ``so_rpy`` one-step prediction and the measured
next state from flight logs:

    delta(x, u) = x_{t+1} - rk4_step(f_nominal, x_t, u_t)

The corrected ("exact") model is then ``nominal + GP_mean``, and the GP variance flags where
the data does not constrain the correction. This is an OFFLINE diagnostic / model-refinement
tool: it does not run inside the real-time MPC.

State x (12): [pos(3), rpy(3), vel(3), drpy(3)]. Input u (4): [roll, pitch, yaw, thrust].
The default targets are the dynamic states (vel + drpy), where model error actually lives;
the kinematic states (pos, rpy) integrate exactly from vel/drpy and carry no model error.
"""

from __future__ import annotations

import numpy as np

from sysid.gp_core import GaussianProcess
from sysid.so_rpy_nominal import CF2X_P250, rk4_step

# Concatenated regressor layout z = [x(0..11), u(12..15)].
#   pos 0,1,2 | rpy 3,4,5 | vel 6,7,8 | drpy 9,10,11 | cmd_rpy 12,13,14 | thrust 15
DYNAMIC_OUTPUTS = (6, 7, 8, 9, 10, 11)  # vel + drpy: where so_rpy can be wrong
# Physically-motivated default features: attitude, velocity, body rates, thrust.
DEFAULT_FEATURES = (3, 4, 5, 6, 7, 8, 9, 10, 11, 15)

_STATE_LABELS = [
    "x", "y", "z", "roll", "pitch", "yaw",
    "vx", "vy", "vz", "droll", "dpitch", "dyaw",
]


class DroneResidualGP:
    """Fits per-dimension GP residuals on top of the nominal so_rpy model."""

    def __init__(
        self,
        dt: float,
        params: dict = CF2X_P250,
        output_dims: tuple[int, ...] = DYNAMIC_OUTPUTS,
        feature_dims: tuple[int, ...] = DEFAULT_FEATURES,
        n_restarts: int = 4,
        seed: int = 0,
    ):
        """Initialize the residual model.

        Args:
            dt: Logging / integration time step, in seconds.
            params: Nominal so_rpy parameters.
            output_dims: State indices whose residual is learned (default: vel + drpy).
            feature_dims: Indices into z = [x, u] used as GP inputs.
            n_restarts: Hyperparameter-optimization restarts per GP.
            seed: RNG seed.
        """
        self._dt = dt
        self._params = params
        self._output_dims = tuple(output_dims)
        self._feature_dims = np.asarray(feature_dims, dtype=int)
        self._gps = {
            d: GaussianProcess(n_restarts=n_restarts, seed=seed + i)
            for i, d in enumerate(self._output_dims)
        }
        self._fitted = False

    def _features(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Build the GP input matrix from states and inputs."""
        z = np.concatenate([np.atleast_2d(X), np.atleast_2d(U)], axis=1)
        return z[:, self._feature_dims]

    def _nominal_next(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Nominal one-step prediction for each (x_t, u_t)."""
        return rk4_step(np.atleast_2d(X), np.atleast_2d(U), self._dt, self._params)

    def residual(self, X_t: np.ndarray, U_t: np.ndarray, X_tp1: np.ndarray) -> np.ndarray:
        """Full next-state residual ``x_{t+1} - nominal_next`` (n, 12)."""
        return np.atleast_2d(X_tp1) - self._nominal_next(X_t, U_t)

    def fit(self, X_t: np.ndarray, U_t: np.ndarray, X_tp1: np.ndarray) -> "DroneResidualGP":
        """Fit one GP per output dimension on the nominal-model residual."""
        feats = self._features(X_t, U_t)
        res = self.residual(X_t, U_t, X_tp1)
        for d in self._output_dims:
            self._gps[d].fit(feats, res[:, d])
        self._fitted = True
        return self

    def predict_residual(self, X: np.ndarray, U: np.ndarray):
        """Predict residual mean and std for the output dims at (X, U).

        Returns:
            ``(mean, std)`` each of shape ``(n, len(output_dims))``.
        """
        if not self._fitted:
            raise RuntimeError("DroneResidualGP.predict_residual called before fit().")
        feats = self._features(X, U)
        means, stds = [], []
        for d in self._output_dims:
            m, s = self._gps[d].predict(feats)
            means.append(m)
            stds.append(s)
        return np.stack(means, axis=1), np.stack(stds, axis=1)

    def corrected_next(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Corrected one-step prediction: nominal + GP residual mean (n, 12)."""
        nominal = self._nominal_next(X, U)
        mean, _ = self.predict_residual(X, U)
        corrected = nominal.copy()
        for j, d in enumerate(self._output_dims):
            corrected[:, d] += mean[:, j]
        return corrected

    def evaluate(self, X_t: np.ndarray, U_t: np.ndarray, X_tp1: np.ndarray) -> dict:
        """Per-dimension one-step RMSE of the nominal vs corrected model.

        Returns a dict with arrays ``nominal_rmse`` and ``corrected_rmse`` over the output
        dims, plus the state labels, so you can see where the GP helps.
        """
        X_tp1 = np.atleast_2d(X_tp1)
        nominal = self._nominal_next(X_t, U_t)
        corrected = self.corrected_next(X_t, U_t)
        dims = list(self._output_dims)
        nom_rmse = np.sqrt(np.mean((X_tp1[:, dims] - nominal[:, dims]) ** 2, axis=0))
        cor_rmse = np.sqrt(np.mean((X_tp1[:, dims] - corrected[:, dims]) ** 2, axis=0))
        return {
            "labels": [_STATE_LABELS[d] for d in dims],
            "nominal_rmse": nom_rmse,
            "corrected_rmse": cor_rmse,
        }
