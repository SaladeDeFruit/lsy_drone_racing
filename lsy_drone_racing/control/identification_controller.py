"""Open-loop excitation controller for offline model-identification flights.

Emits attitude + collective-thrust commands ``[roll, pitch, yaw, thrust]`` that excite a
chosen part of the dynamics, while logging every step to disk via
:class:`sysid.flight_logger.FlightLogger`. Intended to be run in ``control_mode = "attitude"``
so that the logged action IS the so_rpy model input ``u``.

Maneuvers (set ``[identification].maneuver`` in the config, or edit ``_MANEUVER`` below):
``hover``, ``vertical``, ``roll``, ``pitch``, ``yaw``, ``combined``.

WARNING: this is an open-loop excitation SCAFFOLD. It does not stabilize the drone. Validate
amplitudes, add a stabilizing baseline / safety bounds, and tune on your own rig before
flying real hardware. Only the logging path is unit-tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller
from sysid.flight_logger import FlightLogger
from sysid.so_rpy_nominal import CF2X_P250

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MANEUVER = "roll"  # default if the config has no [identification] section
_DURATION = 12.0  # s
_RPY_AMP = 0.2  # rad, attitude excitation amplitude
_THRUST_AMP = 0.15  # fraction of hover thrust for vertical excitation
_CHIRP_F0, _CHIRP_F1 = 0.2, 4.0  # Hz, sweep range


class IdentificationController(Controller):
    """Excitation + logging controller for identification flights."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize excitation parameters and the flight logger."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        ident = config.get("identification", {}) if hasattr(config, "get") else {}
        self._maneuver = ident.get("maneuver", _MANEUVER)
        self._duration = ident.get("duration", _DURATION)
        self._hover_thrust = CF2X_P250["mass"] * -CF2X_P250["gravity_vec"][-1]  # total N
        self._logger = FlightLogger(out_dir="logs", run_name=f"ident_{self._maneuver}")
        self._tick = 0

    def _chirp(self, t: float) -> float:
        """Linear-frequency sine sweep from _CHIRP_F0 to _CHIRP_F1 over the flight."""
        k = (_CHIRP_F1 - _CHIRP_F0) / max(self._duration, 1e-6)
        phase = 2 * np.pi * (_CHIRP_F0 * t + 0.5 * k * t**2)
        return np.sin(phase)

    def _excitation(self, t: float) -> np.ndarray:
        """Return [roll, pitch, yaw, thrust] for the selected maneuver at time t."""
        roll = pitch = yaw = 0.0
        thrust = self._hover_thrust
        s = self._chirp(t)
        if self._maneuver == "hover":
            pass
        elif self._maneuver == "vertical":
            thrust = self._hover_thrust * (1.0 + _THRUST_AMP * s)
        elif self._maneuver == "roll":
            roll = _RPY_AMP * s
        elif self._maneuver == "pitch":
            pitch = _RPY_AMP * s
        elif self._maneuver == "yaw":
            yaw = _RPY_AMP * s
        elif self._maneuver == "combined":
            roll = _RPY_AMP * np.sin(2 * np.pi * 1.1 * t)
            pitch = _RPY_AMP * np.sin(2 * np.pi * 1.7 * t)
            yaw = 0.5 * _RPY_AMP * np.sin(2 * np.pi * 0.9 * t)
        else:
            raise ValueError(f"Unknown maneuver: {self._maneuver}")
        return np.array([roll, pitch, yaw, thrust])

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the excitation command and log the (state, action) pair."""
        t = self._tick / self._freq
        action = self._excitation(t)
        self._logger.log(t=t, obs=obs, action=action)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance time; finish when the excitation duration is reached."""
        self._tick += 1
        return self._tick / self._freq >= self._duration

    def episode_callback(self):
        """Persist the log at the end of the episode."""
        self._logger.save()
        self._tick = 0
