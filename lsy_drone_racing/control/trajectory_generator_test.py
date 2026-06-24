"""Track-aware reference trajectory generation for the racing controllers.

This module replaces the hard-coded, hand-tuned waypoints previously embedded in the MPC
controller (see :mod:`lsy_drone_racing.control.attitude_mpc`) with a trajectory that is
derived from the *actual* track, i.e. the observed gate and obstacle positions.

The generated trajectory passes through every gate center, crosses each gate along its
normal axis (the same gate-frame x-axis convention used by
:func:`lsy_drone_racing.envs.utils.gate_passed`), and keeps a simple safety margin around
the cylindrical obstacles. It is intentionally *not* a time-optimal or collision-free
planner: real-time safety is the job of the MPC constraints, this module only provides a
sensible reference to track.

This is a helper, not a :class:`~lsy_drone_racing.control.controller.Controller`. It is
imported by the controllers, which keeps the "one Controller class per file" rule intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Geometry of the level configs (see config/level0.toml).
GATE_OPENING = 0.4  # m, square gate opening (inner frame dimension)
OBSTACLE_RADIUS = 0.015  # m, obstacles are 0.03 m diameter cylinders


class TrajectoryGenerator:
    """Generate a track-aware reference trajectory from gates and obstacles.

    The trajectory is a cubic spline through the start position and every gate. For each
    gate we additionally insert an approach and an exit waypoint along the gate normal so
    that the spline crosses the gate plane roughly perpendicularly (centered in the
    opening). Waypoints between gates are nudged away from obstacles to keep a safety
    margin. The spline is sampled at ``freq`` Hz to produce the reference arrays consumed
    by the controllers.

    Attributes:
        pos: Sampled positions, shape ``(T, 3)``.
        vel: Sampled velocities (spline derivative), shape ``(T, 3)``.
        yaw: Sampled yaw reference, shape ``(T,)``. Currently always zero; yaw alignment
            will be added here later.
        gate_crossing_idx: Index into ``pos``/``vel`` at which the trajectory crosses each
            gate plane, shape ``(n_gates,)``.
    """

    def __init__(
        self,
        start_pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        freq: float,
        t_total: float = 15.0,
        approach_dist: float = 0.25,
        safety_margin: float = 0.1,
    ):
        """Build the reference trajectory.

        Args:
            start_pos: Initial drone position, shape ``(3,)``.
            gates_pos: Gate center positions, shape ``(n_gates, 3)``.
            gates_quat: Gate orientations as xyzw quaternions, shape ``(n_gates, 4)``.
            obstacles_pos: Obstacle positions, shape ``(n_obstacles, 3)``.
            freq: Sampling frequency of the output arrays, in Hz.
            t_total: Total duration of the trajectory, in seconds.
            approach_dist: Distance of the approach/exit waypoints from a gate center along
                the gate normal, in meters.
            safety_margin: Extra clearance added to the obstacle radius, in meters.
        """
        self._freq = freq
        self._t_total = t_total
        self._gates_pos = np.asarray(gates_pos, dtype=float)
        self._gates_quat = np.asarray(gates_quat, dtype=float)
        self._obstacles_pos = np.asarray(obstacles_pos, dtype=float)
        self._clearance = OBSTACLE_RADIUS + safety_margin

        # Gate normals in world frame: the gate-frame x-axis, expressed in world
        # coordinates (same axis the drone crosses in gate_passed).
        self._gate_normals = R.from_quat(self._gates_quat).apply(np.array([1.0, 0.0, 0.0]))

        self._n_samples = int(freq * t_total)
        waypoints = self._build_waypoints(np.asarray(start_pos, dtype=float), approach_dist)
        self._pos_spline = self._plan(waypoints)
        self._vel_spline = self._pos_spline.derivative()

        t = np.linspace(0.0, t_total, self._n_samples)
        self.pos: NDArray[np.floating] = self._pos_spline(t)
        self.vel: NDArray[np.floating] = self._vel_spline(t)
        self.yaw: NDArray[np.floating] = np.zeros(len(t))
        self.gate_crossing_idx: NDArray[np.intp] = self._find_gate_crossings()

    def _build_waypoints(
        self, start_pos: NDArray[np.floating], approach_dist: float
    ) -> NDArray[np.floating]:
        """Assemble the waypoint list: start, then approach/center/exit per gate."""
        waypoints = [start_pos]
        for center, normal in zip(self._gates_pos, self._gate_normals):
            waypoints.append(center - approach_dist * normal)  # approach (-x gate side)
            waypoints.append(center)  # gate center
            waypoints.append(center + approach_dist * normal)  # exit (+x gate side)
        return np.array(waypoints)

    def _fit(self, waypoints: NDArray[np.floating]) -> CubicSpline:
        """Fit a cubic spline through ``waypoints`` with chord-length parameterization."""
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        self._t_knots = cum / cum[-1] * self._t_total
        return CubicSpline(self._t_knots, waypoints)

    def _plan(
        self, waypoints: NDArray[np.floating], max_iter: int = 8, buffer: float = 0.05
    ) -> CubicSpline:
        """Plan a spline that clears the obstacles by inserting detour waypoints.

        We fit the spline through the gate waypoints, sample it, and find the worst
        obstacle violation in the xy-plane. A detour waypoint is inserted at the point of
        closest approach, pushed radially out to the clearance (plus a small buffer), and
        the spline is refit. This repeats until no violation remains or ``max_iter`` is
        hit. Gate centers are never moved, so traversal is preserved. This is a simple
        reference-shaping pass; the MPC handles real-time obstacle safety.
        """
        t = np.linspace(0.0, self._t_total, self._n_samples)
        target = self._clearance + buffer
        spline = self._fit(waypoints)
        for _ in range(max_iter):
            pos = spline(t)
            worst = None  # (sample_idx, detour_xy)
            worst_dist = self._clearance
            for obs in self._obstacles_pos:
                d_xy = pos[:, :2] - obs[:2]
                dist = np.linalg.norm(d_xy, axis=1)
                k = int(np.argmin(dist))
                if dist[k] < worst_dist:
                    worst_dist = dist[k]
                    direction = d_xy[k] / dist[k] if dist[k] > 1e-6 else np.array([1.0, 0.0])
                    worst = (k, obs[:2] + direction * target)
            if worst is None:
                break
            k, detour_xy = worst
            detour = np.array([detour_xy[0], detour_xy[1], pos[k, 2]])
            # Insert the detour at the matching arc position, never before the start.
            j = int(np.clip(np.searchsorted(self._t_knots, t[k]), 1, len(waypoints)))
            waypoints = np.insert(waypoints, j, detour, axis=0)
            spline = self._fit(waypoints)
        self._waypoints = waypoints
        return spline

    def _find_gate_crossings(self) -> NDArray[np.intp]:
        """Find the sample index where the trajectory crosses each gate plane.

        Uses the same gate-frame convention as
        :func:`lsy_drone_racing.envs.utils.gate_passed`: transform sampled positions into
        the gate frame and detect where the local x-coordinate changes sign from negative
        to positive. The crossing nearest the gate center is selected.
        """
        idxs = np.zeros(len(self._gates_pos), dtype=np.intp)
        for g, (center, quat) in enumerate(zip(self._gates_pos, self._gates_quat)):
            local = R.from_quat(quat).apply(self.pos - center, inverse=True)
            x_local = local[:, 0]
            crossings = np.where((x_local[:-1] < 0) & (x_local[1:] >= 0))[0]
            if len(crossings) == 0:
                # Fallback: closest approach along the normal axis.
                idxs[g] = int(np.argmin(np.abs(x_local)))
            else:
                # Pick the crossing whose in-plane offset is smallest (best centered).
                offset = np.linalg.norm(local[crossings][:, 1:], axis=1)
                idxs[g] = int(crossings[np.argmin(offset)])
        return idxs

    def gate_crossing_offsets(self) -> NDArray[np.floating]:
        """Return the in-plane (y, z) offset from each gate center at crossing.

        Useful for validation: each row should be well within ``GATE_OPENING / 2``.
        """
        offsets = np.zeros((len(self._gates_pos), 2))
        for g, (center, quat) in enumerate(zip(self._gates_pos, self._gates_quat)):
            cross_pos = self.pos[self.gate_crossing_idx[g]]
            local = R.from_quat(quat).apply(cross_pos - center, inverse=True)
            offsets[g] = local[1:]
        return offsets
