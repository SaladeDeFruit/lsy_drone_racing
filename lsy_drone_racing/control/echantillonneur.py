"""Stage 3 of the control pipeline: a pure sampler of a TOPP-RA trajectory.

Pipeline (strict separation of concerns):
  1. Spatial path generation (separate file) -> the geometric path (e.g. a
     ``toppra.SplineInterpolator`` over a normalized arc parameter). The ONLY source of
     geometry (waypoints, gates, obstacles all live there).
  2. TOPP-RA -> time-parameterizes that path under kinematic limits. Produces a trajectory
     object ``traj`` that is callable: ``traj(t, 0)`` position, ``traj(t, 1)`` velocity,
     ``traj(t, 2)`` acceleration, and exposes ``traj.duration``.
  3. THIS module: samples ``traj`` at ``freq`` Hz into reference arrays for the MPC.
  4. MPC (AttitudeMPCRacing): tracks the sampled reference.

This stage contains NO waypoint, gate, obstacle, or spline logic — it only samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryGenerator:
    """Sample a TOPP-RA time trajectory into MPC reference arrays.

    Attributes:
        pos: Sampled positions, shape ``(T, 3)``.
        vel: Sampled velocities, shape ``(T, 3)``.
        acc: Sampled accelerations, shape ``(T, 3)`` (useful for MPC feed-forward).
        yaw: Yaw reference, shape ``(T,)``. **This is identically zero and is NOT meant to
            be tracked.** The MPC weights yaw at 0 in its cost, so the optimizer ignores this
            array entirely. It exists only to fill the reference layout.

            DO NOT repurpose this array as a tangent / heading yaw target. There is no
            external perception on board, so no yaw target is useful; forcing a yaw only
            wastes torque/thrust budget. Keep it zero.
        duration: Trajectory duration in seconds (mirrors ``traj.duration``).
    """

    def __init__(self, toppra_trajectory, freq: float):
        """Sample the given TOPP-RA trajectory.

        Args:
            toppra_trajectory: The trajectory object returned by
                ``toppra_instance.compute_trajectory()`` (stage 2). Must be callable as
                ``traj(t, order)`` (order 0/1/2 -> pos/vel/acc) on an array ``t`` and expose
                ``duration``. This sampler never computes TOPP-RA itself.
            freq: Sampling frequency in Hz.
        """
        traj = toppra_trajectory
        self.duration: float = float(traj.duration)
        # Single vectorized sampling call per derivative — no Python loop.
        n_samples = max(int(freq * self.duration), 2)
        t = np.linspace(0.0, self.duration, n_samples)
        self.pos: NDArray[np.floating] = np.asarray(traj(t, 0), dtype=float)
        self.vel: NDArray[np.floating] = np.asarray(traj(t, 1), dtype=float)
        self.acc: NDArray[np.floating] = np.asarray(traj(t, 2), dtype=float)
        # Yaw is intentionally free / zero (see class docstring). Never use as a target.
        self.yaw: NDArray[np.floating] = np.zeros(n_samples)
