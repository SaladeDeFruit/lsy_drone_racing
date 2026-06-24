"""Stage 4: real-time attitude MPC tracking a sampled TOPP-RA reference.

Pipeline (strict separation of concerns):
  1. Spatial path generation (separate file, NOT here) -> geometric path over a normalized
     arc parameter (a ``toppra.SplineInterpolator`` or equivalent callable).
  2. TOPP-RA -> time-parameterizes that path under per-axis kinematic limits ->
     ``traj = instance.compute_trajectory()`` (callable: ``traj(t,0/1/2)`` = pos/vel/acc,
     with ``traj.duration``).
  3. TrajectoryGenerator -> pure sampler of ``traj`` into reference arrays.
  4. THIS MPC -> tracks the sampled reference with state feedback; outputs a Cartesian
     ``"state"`` command.

Yaw is left FREE (zero cost weight): there is no onboard perception that would benefit from a
heading target, so forcing yaw would only waste torque/thrust budget.
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
from lsy_drone_racing.control.echantillonneur import TrajectoryGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

# State layout: [pos(0:3), rpy(3:6), vel(6:9), drpy(9:12)]. yaw = index 5, yaw-rate = 11.
YAW_IDX = 5

# Default TOPP-RA kinematic limits (configurable, per-axis, asymmetric). Format: (dof, 2) =
# [lower, upper]. z is asymmetric: gravity lets the drone decelerate downward harder than it
# can accelerate upward.
DEFAULT_VEL_LIMIT = np.array([[-1.8, 1.8], [-1.8, 1.8], [-1.0, 1.0]])  # m/s
DEFAULT_ACC_LIMIT = np.array([[-4.5, 4.5], [-4.5, 4.5], [-8.2, 4.5]])  # m/s^2


def _wrap_to_pi(angle):
    """Wrap an angle (or array) to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    """Creates the racing OCP and solver (SQP_RTI, plain tracking, FREE yaw)."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    # yaw (index 5) weight = 0 -> yaw is FREE. A small weight is kept on the yaw rate
    # (index 11) to damp abrupt nose rotations.
    Q = np.diag(
        [
            50.0, 50.0, 400.0,  # pos
            1.0, 1.0, 0.0,  # rpy: roll, pitch tracked; yaw FREE (weight 0)
            10.0, 10.0, 10.0,  # vel
            5.0, 5.0, 2.0,  # drpy: small reg on yaw rate (index 11)
        ]
    )
    R_in = np.diag([1.0, 1.0, 1.0, 50.0])  # rpy commands + thrust
    Q_e = Q.copy()  # terminal cost inherits the free-yaw weighting
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

    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
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
    """Real-time MPC tracking a sampled TOPP-RA reference, outputting a state command."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and build the sampled reference (stages 1->3)."""
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Stages 1 + 2 + 3 -> sampled reference arrays.
        traj_gen = self._build_reference(obs, config)
        self._waypoints_pos = traj_gen.pos
        self._waypoints_vel = traj_gen.vel
        self._waypoints_yaw = traj_gen.yaw  # zeros; NOT tracked (yaw is free)
        self._duration = traj_gen.duration

        # Adaptive indexing: reference length depends on traj.duration (variable), not a fixed
        # 15 s. Guard against a trajectory too short for the horizon.
        self._T = len(self._waypoints_pos)
        if self._T < self._N + 1:
            raise ValueError(
                f"Sampled trajectory too short ({self._T} samples) for horizon N={self._N}. "
                f"Need >= N+1 samples: lower N, raise freq, or lengthen the trajectory."
            )

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = self._T - 1 - self._N
        self._config = config
        self._finished = False

    def _build_reference(self, obs: dict, config: dict) -> TrajectoryGenerator:
        """Build the sampled reference: spatial path (1) -> TOPP-RA (2) -> sampler (3).

        Stage 1 lives in a SEPARATE module (not created here). It must return the geometric
        path as a toppra-compatible interpolator (see the accompanying note for the contract).
        We only feed that path to TOPP-RA with configurable, per-axis asymmetric kinematic
        limits, then wrap the resulting trajectory in the pure sampler.
        """
        # Lazy imports: stage 1 and toppra are external to this real-time module.
        from lsy_drone_racing.control.spatial_path import build_spatial_path  # stage 1
        import toppra.algorithm as toppra_algo
        import toppra.constraint as toppra_constraint

        path = build_spatial_path(obs, config)  # stage 1: geometry only

        tp = config.get("toppra", {}) if hasattr(config, "get") else {}
        vlim = np.asarray(tp.get("vel_limit", DEFAULT_VEL_LIMIT), dtype=float)
        alim = np.asarray(tp.get("acc_limit", DEFAULT_ACC_LIMIT), dtype=float)
        pc_vel = toppra_constraint.JointVelocityConstraint(vlim)
        pc_acc = toppra_constraint.JointAccelerationConstraint(alim)
        instance = toppra_algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        )
        traj = instance.compute_trajectory()
        if traj is None:
            raise RuntimeError(
                "TOPP-RA failed to compute a trajectory. Check the stage-1 path and limits."
            )
        return TrajectoryGenerator(traj, config.env.freq)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next Cartesian state command.

        Returns ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]``. The yaw field
        is made continuous w.r.t. the measured yaw to avoid a +/-pi wrap discontinuity.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw (zero, weight 0 -> ignored)
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

        # Optimized next state (horizon stage 1) -> Cartesian "state" command.
        x1 = self._acados_ocp_solver.get(1, "x")
        pos1, rpy1, vel1, drpy1 = x1[0:3], x1[3:6], x1[6:9], x1[9:12]
        acc = (vel1 - x0[6:9]) / self._dt
        # Yaw continuity: commanded yaw = continuous value nearest the measured yaw, so a wrap
        # across +/-pi never produces a discontinuous setpoint downstream.
        yaw_meas = x0[YAW_IDX]
        yaw_cmd = yaw_meas + _wrap_to_pi(rpy1[2] - yaw_meas)
        return np.concatenate((pos1, vel1, acc, [yaw_cmd], drpy1))

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
