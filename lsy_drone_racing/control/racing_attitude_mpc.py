"""Racing attitude MPC: tracks the TOGT -> TOPP-RA trajectory and outputs ATTITUDE commands.

Control mode: ``attitude``. The action returned is ``[roll, pitch, yaw, collective_thrust]``
(see crazyflow ``control/control.py`` and ``race_core.build_action_space``).

Pipeline:
  1. Reference: ``togt_optimized_waypoints`` (TOGT optimizer, gates in the combinations-optimal
     order) -> chord-length ``SplineInterpolator`` -> TOPP-RA -> ``TrajectoryGenerator`` sampler.
  2. MPC (acados, ``so_rpy`` model, inputs = roll/pitch/yaw/thrust): tracks the sampled
     position/velocity reference. Roll AND pitch emerge from tracking (natural banking).
  3. Risky-gate roll maneuver: at gates flagged in ``RISKY_GATE_ROLL`` the per-stage cost is
     reshaped over a short window around the gate crossing to FORCE a large roll (knife-edge),
     while relaxing the position weight (the bank kills lift -> a brief altitude dip is allowed).
     Injected purely through the cost so the MPC stays in closed loop (no output override).

Yaw stays FREE (weight 0): the env's attitude yaw command is bounded to +/-pi/2, so a gate
heading near +/-pi could not be commanded anyway; we do not force yaw.
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
from lsy_drone_racing.control.combinations_path_generator import chord_length_param
from lsy_drone_racing.control.echantillonneur import TrajectoryGenerator
from lsy_drone_racing.control.pid_controller import togt_optimized_waypoints

if TYPE_CHECKING:
    from numpy.typing import NDArray

# State layout: [pos(0:3), rpy(3:6)=roll,pitch,yaw, vel(6:9), drpy(9:12)].
ROLL_IDX, PITCH_IDX, YAW_IDX = 3, 4, 5

# Per-axis TOPP-RA kinematic limits for the reference. Format (dof, 2) = [lower, upper].
# Kept moderate so the attitude MPC tracks the reference smoothly (an over-aggressive,
# time-optimal reference makes the MPC saturate thrust and chatter the roll command).
DEFAULT_VEL_LIMIT = np.array([[-1.5, 1.5], [-1.5, 1.5], [-0.8, 0.8]])  # m/s
DEFAULT_ACC_LIMIT = np.array([[-2.5, 2.5], [-2.5, 2.5], [-4, 3]])  # m/s^2

# HOOK: another module may set this (before the controller is instantiated) to flag risky gates
# and the roll angle (radians) to force at each. Key = gate index in ``obs["gates_pos"]``.
# Example: ``racing_attitude_mpc.RISKY_GATE_ROLL = {2: np.pi / 2}``.
RISKY_GATE_ROLL: dict[int, float] = {}

# Roll-maneuver tuning (knobs).
ROLL_WINDOW_S = 0.15  # half-width of the roll window, in seconds
ROLL_WEIGHT_WINDOW = 200.0  # roll state weight inside the window (base is 1.0)
POS_WEIGHT_WINDOW = np.array([10.0, 10.0, 5.0])  # reduced pos weight (esp. z) inside the window
 

def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic so_rpy drone model (attitude inputs)."""
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
    model.name = "racing_attitude_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    return model


def base_cost_weights() -> tuple[NDArray, NDArray]:
    """Base state/input cost diagonals (Q, R). Roll/pitch tracked lightly; yaw FREE (weight 0)."""
    Q = np.diag(
        [
            50.0, 50.0, 400.0,  # pos
            0, 0, 0.0,  # rpy: roll, pitch tracked; yaw FREE (weight 0)
            10.0, 10.0, 10.0,  # vel
            5.0, 5.0, 2.0,  # drpy
        ]
    )
    R_in = np.diag([0.5, 0.5, 0.5, 50.0])  # roll/pitch/yaw commands + thrust
    return Q, R_in


def create_ocp_solver(
    Tf: float,
    N: int,
    parameters: dict,
    roll_bound: float = 0.5,
    soft_rpy: bool = False,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates the racing attitude OCP/solver.

    Args:
        roll_bound: hard bound on the roll state/input (rad). 0.5 for plain banking (matches the
            stable ``attitude_mpc``); widened to ~pi/2 only when the knife-edge maneuver is used.
        soft_rpy: if True, slack the rpy box (needed when the maneuver pushes near the bound).
            Off for plain banking (the extra slack variables make the roll command chatter).
    """
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Q, R_in = base_cost_weights()
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

    # State box: roll widened to roll_bound (knife-edge), pitch/yaw kept tight.
    ocp.constraints.lbx = np.array([-roll_bound, -0.5, -0.5])
    ocp.constraints.ubx = np.array([roll_bound, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([ROLL_IDX, PITCH_IDX, YAW_IDX])
    if soft_rpy:
        ocp.constraints.idxsbx = np.array([0, 1, 2])  # slack roll, pitch, yaw bounds
        ns = 3
        ocp.cost.zl = 1e3 * np.ones(ns)
        ocp.cost.zu = 1e3 * np.ones(ns)
        ocp.cost.Zl = 1e2 * np.ones(ns)
        ocp.cost.Zu = 1e2 * np.ones(ns)

    # Input box: roll command widened to roll_bound; the env attitude action allows +/-pi/2.
    ocp.constraints.lbu = np.array([-roll_bound, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([roll_bound, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.levenberg_marquardt = 1e-3
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/racing_attitude_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return acados_ocp_solver, ocp


class RacingAttitudeMPC(Controller):
    """Attitude MPC tracking the TOGT->TOPP-RA reference, with forced roll at risky gates."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the reference, the solver, and the per-sample roll-maneuver profiles."""
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Reference: TOGT -> TOPP-RA -> sampled arrays.
        traj_gen = self._build_reference(obs, config)
        self._waypoints_pos = traj_gen.pos
        self._waypoints_vel = traj_gen.vel
        self._waypoints_acc = traj_gen.acc
        self._duration = traj_gen.duration
        self._T = len(self._waypoints_pos)
        if self._T < self._N + 1:
            raise ValueError(
                f"Sampled trajectory too short ({self._T}) for horizon N={self._N}."
            )

        # Risky-gate maneuver active? (decides the solver's roll bound / slack.)
        # Sources, merged: module-level RISKY_GATE_ROLL (hardcoded default, editable in this
        # file) overridden by config ``risky_gates`` (robust hook: another module / the config
        # can set {gate_idx: roll_rad} without depending on this module's global identity, which
        # ``load_controller`` re-creates). Keys normalized to int, values to float radians.
        self._risky_gates = {int(k): float(v) for k, v in RISKY_GATE_ROLL.items()}
        cfg_risky = config.get("risky_gates", {}) if hasattr(config, "get") else {}
        self._risky_gates.update({int(k): float(v) for k, v in dict(cfg_risky).items()})
        self._has_maneuver = len(self._risky_gates) > 0

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON,
            self._N,
            self.drone_params,
            roll_bound=(np.pi / 2 if self._has_maneuver else 0.5),
            soft_rpy=self._has_maneuver,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # Per-sample roll-maneuver profiles (default: no maneuver = plain banking tracking).
        self._build_roll_profiles(obs, config)

        # Base cost matrices used to rebuild per-stage W only where the maneuver is active.
        Q, R_in = base_cost_weights()
        self._base_W = scipy.linalg.block_diag(Q, R_in)
        self._base_W_e = Q.copy()

        self._tick = 0
        self._tick_max = self._T - 1 - self._N
        self._config = config
        self._finished = False

        self._warm_start(0)

    def _build_reference(self, obs: dict, config: dict) -> TrajectoryGenerator:
        """TOGT waypoints -> chord-length spline -> TOPP-RA -> sampler."""
        import toppra as ta
        import toppra.algorithm as toppra_algo
        import toppra.constraint as toppra_constraint

        waypoints = togt_optimized_waypoints(obs["pos"], obs["gates_pos"], obs["gates_quat"], exit_dist=0.3, gate_use_frac=0.1)
        ss = chord_length_param(waypoints)
        path = ta.SplineInterpolator(ss, waypoints)

        pc_vel = toppra_constraint.JointVelocityConstraint(DEFAULT_VEL_LIMIT)
        pc_acc = toppra_constraint.JointAccelerationConstraint(DEFAULT_ACC_LIMIT)
        instance = toppra_algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        )
        traj = instance.compute_trajectory()
        if traj is None:
            raise RuntimeError("TOPP-RA failed. Check the TOGT waypoints and limits.")
        return TrajectoryGenerator(traj, config.env.freq)

    def _build_roll_profiles(self, obs: dict, config: dict) -> None:
        """Per-sample roll reference / roll weight / position weight for the risky-gate maneuver.

        For each gate flagged in ``RISKY_GATE_ROLL``, the trajectory sample nearest the gate
        center marks the maneuver center; a smooth half-cosine bump over a +/-ROLL_WINDOW_S
        window forces the roll up to the requested angle, boosts the roll weight, and relaxes the
        position weight so the inevitable altitude dip is permitted instead of fought.
        """
        T = self._T
        self._roll_ref = np.zeros(T)
        self._roll_w = np.full(T, 1.0)  # base roll weight (matches Q[ROLL_IDX])
        self._pos_w = np.tile(np.array([50.0, 50.0, 400.0]), (T, 1))  # base pos weights

        if not self._has_maneuver:
            return

        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        win = max(1, int(ROLL_WINDOW_S * config.env.freq))
        for gate_idx, angle in self._risky_gates.items():
            center = gates_pos[gate_idx]
            k_gate = int(np.argmin(np.linalg.norm(self._waypoints_pos - center, axis=1)))
            lo, hi = max(0, k_gate - win), min(T - 1, k_gate + win)
            for k in range(lo, hi + 1):
                # Half-cosine bump: 1 at k_gate, 0 at the window edges.
                bump = 0.5 * (1.0 + np.cos(np.pi * (k - k_gate) / win))
                if bump <= self._roll_ref[k] / (angle + 1e-9):
                    continue  # keep the stronger maneuver if windows overlap
                self._roll_ref[k] = angle * bump
                self._roll_w[k] = ROLL_WEIGHT_WINDOW
                self._pos_w[k] = POS_WEIGHT_WINDOW

    def _warm_start(self, i: int) -> None:
        """Seed the solver with the reference states and hover thrust (avoids cold-start NaN)."""
        u_hover = np.array([0.0, 0.0, 0.0, self._hover_thrust])
        for j in range(self._N + 1):
            k = min(i + j, self._T - 1)
            xg = np.zeros(self._nx)
            xg[0:3] = self._waypoints_pos[k]
            xg[6:9] = self._waypoints_vel[k]
            self._acados_ocp_solver.set(j, "x", xg)
            if j < self._N:
                self._acados_ocp_solver.set(j, "u", u_hover)

    def _stage_weight(self, k: int, terminal: bool) -> NDArray:
        """Base cost matrix with pos/roll diagonals overridden by the maneuver profiles."""
        W = (self._base_W_e if terminal else self._base_W).copy()
        W[0, 0], W[1, 1], W[2, 2] = self._pos_w[k]
        W[ROLL_IDX, ROLL_IDX] = self._roll_w[k]
        return W

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Solve the OCP and return the attitude command ``[roll, pitch, yaw, thrust]``."""
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        for j in range(self._N):
            k = min(i + j, self._T - 1)
            yref = np.zeros(self._ny)
            yref[0:3] = self._waypoints_pos[k]
            yref[ROLL_IDX] = self._roll_ref[k]  # roll reference (0 unless maneuver active)
            yref[6:9] = self._waypoints_vel[k]
            yref[15] = self._hover_thrust
            self._acados_ocp_solver.set(j, "yref", yref)
            if self._has_maneuver:
                self._acados_ocp_solver.cost_set(j, "W", self._stage_weight(k, terminal=False))

        k_e = min(i + self._N, self._T - 1)
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self._waypoints_pos[k_e]
        yref_e[ROLL_IDX] = self._roll_ref[k_e]
        yref_e[6:9] = self._waypoints_vel[k_e]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)
        if self._has_maneuver:
            self._acados_ocp_solver.cost_set(self._N, "W", self._stage_weight(k_e, terminal=True))

        status = self._acados_ocp_solver.solve()
        if status != 0:
            self._warm_start(i)
            status = self._acados_ocp_solver.solve()

        u0 = self._acados_ocp_solver.get(0, "u")  # [roll, pitch, yaw, collective_thrust]
        if status != 0 or not np.all(np.isfinite(u0)):
            # Safe fallback: level attitude + hover thrust (never emit a NaN attitude command).
            return np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float32)
        return np.asarray(u0, dtype=np.float32)

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
