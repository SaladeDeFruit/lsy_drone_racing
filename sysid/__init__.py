"""Offline drone-model system identification (GP residual on top of nominal so_rpy)."""

from sysid.gp_core import GaussianProcess
from sysid.residual_gp import DEFAULT_FEATURES, DYNAMIC_OUTPUTS, DroneResidualGP
from sysid.so_rpy_nominal import CF2X_P250, rk4_step, so_rpy_xdot

__all__ = [
    "GaussianProcess",
    "DroneResidualGP",
    "DYNAMIC_OUTPUTS",
    "DEFAULT_FEATURES",
    "CF2X_P250",
    "rk4_step",
    "so_rpy_xdot",
]
