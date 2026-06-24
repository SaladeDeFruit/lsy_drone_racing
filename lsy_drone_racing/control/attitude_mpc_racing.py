"""Racing-oriented attitude MPC (plain tracking, no gate/obstacle constraints).

This is an evolved copy of :mod:`lsy_drone_racing.control.attitude_mpc` (kept untouched as
the pedagogical reference). The differences versus the example MPC are:

1. Solver: ``SQP_RTI`` (real-time iteration) instead of full ``SQP`` so the per-cycle
   computation time is bounded (budget ~20 ms at ``config.env.freq = 50 Hz``).
2. Reference: a track-aware :class:`TrajectoryGenerator` (built from the observed gates and
   obstacles) replaces the hard-coded waypoints. Gate traversal and obstacle clearance are
   handled upstream, in the trajectory, not by MPC constraints.
3. Output: a Cartesian ``"state"`` command (see :meth:`compute_control`).

Out of scope here: gate/obstacle MPC constraints, yaw alignment, MPCC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

# The trajectory generator lives in this file in the current tree (renamed from
# trajectory.py). Import from the actual module name.
from lsy_drone_racing.control.trajectory_generator_test import TrajectoryGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic so_rpy drone model."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "attitude_mpc_racing"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates the racing OCP and solver (SQP_RTI, plain trajectory tracking)."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # --- Cost (identical to the example MPC) ---
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Q = np.diag(
        [
            50.0, 50.0, 400.0,  # pos
            1.0, 1.0, 1.0,  # rpy
            10.0, 10.0, 10.0,  # vel
            5.0, 5.0, 5.0,  # drpy
        ]
    )
    R_in = np.diag([1.0, 1.0, 1.0, 50.0])  # rpy commands + thrust
    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R_in)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # --- State / input box constraints (rpy < ~30 deg, thrust limits) ---
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    # --- Solver options ---
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # bounded compute per cycle
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_racing_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return acados_ocp_solver, ocp


class AttitudeMPCRacing(Controller):
    """Attitude MPC with a track-aware reference, outputting a Cartesian state command."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the racing MPC controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Track-aware reference from the observed gates and obstacles. Gate/obstacle
        # avoidance is handled here (upstream), not by the MPC.
        self._traj = TrajectoryGenerator(
            start_pos=obs["pos"],
            gates_pos=np.asarray(obs["gates_pos"], dtype=float),
            gates_quat=np.asarray(obs["gates_quat"], dtype=float),
            obstacles_pos=np.asarray(obs["obstacles_pos"], dtype=float),
            freq=config.env.freq,
        )
        self._waypoints_pos = self._traj.pos
        self._waypoints_vel = self._traj.vel
        self._waypoints_yaw = self._traj.yaw

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired Cartesian state command.

        Returns the MPC's optimized next state (horizon stage 1) reformatted as the
        ``control_mode = "state"`` command expected by the environment:
        ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]``.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Initial state.
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # State / input references.
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw (zero)
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        self._acados_ocp_solver.solve()

        # Optimized next state (stage 1) -> Cartesian "state" command.
        x1 = self._acados_ocp_solver.get(1, "x")
        pos1, rpy1, vel1, drpy1 = x1[0:3], x1[3:6], x1[6:9], x1[9:12]
        # Feed-forward acceleration over the first shooting interval (vel0 = measured).
        acc = (vel1 - x0[6:9]) / self._dt
        return np.concatenate((pos1, vel1, acc, [rpy1[2]], drpy1))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the tick counter."""
        self._tick = 0
