"""Numpy reimplementation of the drone_models ``so_rpy`` nominal dynamics.

This mirrors :func:`drone_models.so_rpy.symbolic_dynamics_euler` so the offline system
identification can compute the nominal one-step prediction without requiring the
``drone_models`` package (which is not always installed). The equations are:

    pos_dot = vel
    rpy_dot = drpy
    vel_dot = R(rpy) @ [0, 0, acc_coef + cmd_f_coef * thrust] / mass + gravity_vec
    ddrpy   = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy

State x (12): [pos(3), rpy(3), vel(3), drpy(3)]. Input u (4): [roll, pitch, yaw, thrust].
Note: the inertia J does NOT appear here — the rotational dynamics are a fitted, per-axis
second-order linear model.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Nominal parameters for the cf2x_P250 (from drone_models so_rpy/params.toml).
CF2X_P250: dict = {
    "mass": 0.0318,
    "gravity_vec": np.array([0.0, 0.0, -9.81]),
    "acc_coef": 0.0,
    "cmd_f_coef": 0.98275823,
    "rpy_coef": np.array([-319.14, -319.14, -284.28]),
    "rpy_rates_coef": np.array([-20.85, -20.85, -38.43]),
    "cmd_rpy_coef": np.array([263.30, 263.30, 502.58]),
}


def so_rpy_xdot(x: np.ndarray, u: np.ndarray, params: dict = CF2X_P250) -> np.ndarray:
    """State derivative of the nominal so_rpy model (batched).

    Args:
        x: State, shape ``(12,)`` or ``(n, 12)``.
        u: Input, shape ``(4,)`` or ``(n, 4)``.
        params: Model parameters dict.

    Returns:
        ``x_dot`` with the same leading shape as ``x``.
    """
    x_in = np.asarray(x, dtype=float)
    single = x_in.ndim == 1
    x = np.atleast_2d(x_in)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    rpy, vel, drpy = x[:, 3:6], x[:, 6:9], x[:, 9:12]
    thrust = u[:, 3]

    rot = R.from_euler("xyz", rpy).as_matrix()  # (n, 3, 3), body->world
    f_body = np.zeros((x.shape[0], 3))
    f_body[:, 2] = params["acc_coef"] + params["cmd_f_coef"] * thrust
    vel_dot = np.einsum("nij,nj->ni", rot, f_body) / params["mass"] + params["gravity_vec"]
    ddrpy = (
        params["rpy_coef"] * rpy
        + params["rpy_rates_coef"] * drpy
        + params["cmd_rpy_coef"] * u[:, 0:3]
    )
    x_dot = np.concatenate([vel, drpy, vel_dot, ddrpy], axis=1)
    return x_dot[0] if single else x_dot


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, params: dict = CF2X_P250) -> np.ndarray:
    """One RK4 integration step of the nominal model (input held constant over ``dt``).

    Args:
        x: State, shape ``(12,)`` or ``(n, 12)``.
        u: Input, shape ``(4,)`` or ``(n, 4)``.
        dt: Time step, in seconds.
        params: Model parameters dict.

    Returns:
        Next state, same leading shape as ``x``.
    """
    k1 = so_rpy_xdot(x, u, params)
    k2 = so_rpy_xdot(x + 0.5 * dt * k1, u, params)
    k3 = so_rpy_xdot(x + 0.5 * dt * k2, u, params)
    k4 = so_rpy_xdot(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
