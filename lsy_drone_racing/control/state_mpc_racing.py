"""Real-time attitude MPC tracking the same TOPP-RA trajectory as the StateController.

Pipeline:
  1. Spatial path -> the fixed race waypoints (with the start at the measured position) define
     a ``toppra.SplineInterpolator`` over a normalized arc parameter. These waypoints and the
     kinematic limits are identical to ``StateController`` (``pid_controller.py``), so this MPC
     tracks the exact same reference the PID-style controller would fly.
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

# Default TOPP-RA kinematic limits (configurable, per-axis). Format: (dof, 2) = [lower,
# upper]. These mirror the StateController (pid_controller.py) limits so this MPC tracks the
# exact same TOPP-RA trajectory the PID-style controller would fly.
DEFAULT_VEL_LIMIT = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.0, 1.0]])  # m/s
DEFAULT_ACC_LIMIT = np.array([[-3.5, 3.5], [-3.5, 3.5], [-2.0, 2.0]])  # m/s^2

# Race waypoints identical to StateController.__init__ (pid_controller.py). The first
# waypoint is replaced at runtime by the measured start position.
RACE_WAYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # placeholder for start_pos = obs["pos"]
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
    # Soften the rpy box into slacked constraints. The aggressive TOPP-RA reference can demand
    # a tilt momentarily beyond +/-0.5 rad (e.g. diagonal corners where x and y accelerate at
    # once); with a HARD box that makes the QP infeasible and HPIPM diverges to NaN (status 3).
    # A heavily penalized slack lets the MPC briefly exceed the limit instead of failing.
    ocp.constraints.idxsbx = np.array([0, 1, 2])  # slack roll, pitch, yaw bounds
    ns = 3
    ocp.cost.zl = 1e3 * np.ones(ns)  # L1 slack penalty (lower)
    ocp.cost.zu = 1e3 * np.ones(ns)  # L1 slack penalty (upper)
    ocp.cost.Zl = 1e2 * np.ones(ns)  # L2 slack penalty (lower)
    ocp.cost.Zu = 1e2 * np.ones(ns)  # L2 slack penalty (upper)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    # Small Levenberg-Marquardt term: regularizes the Gauss-Newton Hessian so a single RTI
    # step from an imperfect warm start stays well-conditioned (avoids HPIPM NaN / status 3).
    ocp.solver_options.levenberg_marquardt = 1e-3
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
        self._waypoints_acc = traj_gen.acc  # smooth feed-forward acceleration for the command
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

        # Hover thrust (collective): cancels gravity at level attitude. Used both as the input
        # reference and to seed the solver so the first RTI step linearizes around a near-feasible
        # guess instead of a zero-thrust free fall.
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        self._tick = 0
        self._tick_max = self._T - 1 - self._N
        self._config = config
        self._finished = False

        # Seed the solver trajectory before the first solve (see _warm_start).
        self._warm_start(0)

    def _warm_start(self, i: int) -> None:
        """Seed the solver's state/input trajectory with the reference and hover thrust.

        SQP_RTI takes a single Newton step per call, so a poor initial guess is never recovered
        within one tick. A zero initial guess means zero collective thrust -> the predicted drone
        free-falls over the horizon and the Gauss-Newton linearization diverges to NaN (HPIPM
        status 3). Seeding hover thrust and the reference positions/velocities gives a sensible,
        near-feasible starting point. Also used to re-seed after a failed solve so one bad QP does
        not poison the warm start of every subsequent tick.
        """
        u_hover = np.array([0.0, 0.0, 0.0, self._hover_thrust])
        for j in range(self._N + 1):
            k = min(i + j, self._T - 1)
            xg = np.zeros(self._nx)
            xg[0:3] = self._waypoints_pos[k]
            xg[6:9] = self._waypoints_vel[k]
            self._acados_ocp_solver.set(j, "x", xg)
            if j < self._N:
                self._acados_ocp_solver.set(j, "u", u_hover)

    def _build_reference(self, obs: dict, config: dict) -> TrajectoryGenerator:
        """Build the sampled reference matching the StateController (pid_controller) trajectory.

        Reproduces ``StateController.__init__``: the same fixed race waypoints (with the start
        replaced by the measured position) define a ``SplineInterpolator``, TOPP-RA
        time-parameterizes it under the same per-axis kinematic limits, and the result is
        wrapped in the pure sampler. This MPC therefore tracks exactly the trajectory the
        PID-style controller would fly. The limits stay overridable via a ``[toppra]`` config
        section, defaulting to the StateController values.
        """
        # Lazy imports: toppra is external to this real-time module.
        import toppra as ta
        import toppra.algorithm as toppra_algo
        import toppra.constraint as toppra_constraint

        # Same waypoints as StateController, but start at the measured drone position.
        waypoints = RACE_WAYPOINTS.copy()
        waypoints[0] = obs["pos"]

        # Geometric path over a normalized arc parameter (identical to StateController).
        ss = np.linspace(0.0, 1.0, len(waypoints))
        path = ta.SplineInterpolator(ss, waypoints)

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
                "TOPP-RA failed to compute a trajectory. Check the waypoints and limits."
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
        yref[:, 15] = self._hover_thrust  # collective thrust reference (gravity compensation)
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solve. On failure (HPIPM status != 0, e.g. NaN/status 3), re-seed the warm start and
        # retry once so a single bad QP does not poison every following tick.
        status = self._acados_ocp_solver.solve()
        if status != 0:
            self._warm_start(i)
            status = self._acados_ocp_solver.solve()

        # Command the TOPP-RA reference target directly (position/velocity/acceleration at the
        # current time index). This is what actually makes the drone FOLLOW the trajectory: the
        # onboard "state" interface flies hard toward the real target, exactly like the
        # StateController (pid_controller.py) which tracks 4 gates. Commanding the MPC's
        # one-step-ahead predicted state instead made the cascade lag (the setpoint sat just
        # ahead of the lagging drone, so it never caught up and crawled along the floor).
        #
        # The MPC solve above remains as a dynamic-feasibility guard: if it reports the reference
        # is infeasible from the current state (NaN / status != 0), we fall back to a zero-accel,
        # measured-yaw setpoint to stay safe instead of pushing a divergent command.
        k = min(i, self._T - 1)
        if status != 0 or not np.all(np.isfinite(self._acados_ocp_solver.get(1, "x"))):
            return np.concatenate(
                (self._waypoints_pos[k], self._waypoints_vel[k], np.zeros(3), [x0[YAW_IDX]], np.zeros(3))
            )
        return np.concatenate(
            (self._waypoints_pos[k], self._waypoints_vel[k], self._waypoints_acc[k], [0.0], np.zeros(3))
        )

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
