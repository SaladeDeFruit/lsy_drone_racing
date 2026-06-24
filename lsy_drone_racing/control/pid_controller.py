from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined TOPP-RA trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Suppress verbose TOPP-RA logging (optional)
        ta.setup_logging("WARNING")

        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

        # 1. Create a geometric path from the waypoints
        # The path parameter 's' ranges from 0 to 1
        ss = np.linspace(0, 1, len(waypoints))
        path = ta.SplineInterpolator(ss, waypoints)

        # 2. Define kinematic constraints for the drone
        # You will need to tune these values based on your drone's physical limits
        v_max_xy = 3.5 
        v_max_z = 2.0
        vbounds = np.array([
            [-v_max_xy, v_max_xy], # X
            [-v_max_xy, v_max_xy], # Y
            [-v_max_z, v_max_z]    # Z
        ])
        
        # Realistic Racing Acceleration Limits (m/s^2)
        # X, Y: Capped around 6 m/s^2 to limit extreme pitching
        # Z: Asymmetrical (-5 downward, +7 upward)
        a_max_xy = 6.0 
        abounds = np.array([
            [-a_max_xy, a_max_xy], # X
            [-a_max_xy, a_max_xy], # Y
            [-5.0, 7.0]            # Z (Negative is down, Positive is up)
        ])
        
        pc_vel = constraint.JointVelocityConstraint(vbounds)
        pc_acc = constraint.JointAccelerationConstraint(abounds)

        # 3. Setup and solve the TOPP-RA problem
        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
        self._trajectory = instance.compute_trajectory()

        if self._trajectory is None:
            raise RuntimeError("TOPP-RA failed to compute a valid trajectory. Check your waypoints and constraints.")

        # Total duration is dynamically computed by TOPP-RA
        self._t_total = self._trajectory.duration
        print(self._t_total)
        print(f"Computed TOPP-RA trajectory with optimal duration: {self._t_total:.2f} s")

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        # Evaluate the TOPP-RA trajectory at the current time t
        # order=0 -> Position, order=1 -> Velocity, order=2 -> Acceleration
        des_pos = self._trajectory(t, 0)
        des_vel = self._trajectory(t, 1)
        des_acc = self._trajectory(t, 2)

        # The drone state array requires 13 elements:
        # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
        # We now pass the dynamically feasible velocity and acceleration as well!
        yaw_and_rates = np.zeros(4) 
        action = np.concatenate((des_pos, des_vel, des_acc, yaw_and_rates), dtype=np.float32)
        
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
        """Increment the time step counter."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        current_time = min(self._tick / self._freq, self._t_total)
        
        # Current Setpoint
        setpoint = self._trajectory(current_time, 0).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        
        # Render the full optimal trajectory
        t_samples = np.linspace(0, self._t_total, 100)
        trajectory_points = np.array([self._trajectory(t, 0) for t in t_samples])
        draw_line(sim, trajectory_points, rgba=(0.0, 1.0, 0.0, 1.0))