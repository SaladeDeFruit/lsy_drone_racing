"""
MPCC Controller for Drone Racing

Features:
- Model Predictive Contouring Control (MPCC) using acados
- Modular path planning via path_planning module
- Dynamic replanning when environment changes
- Configurable speed/stability trade-offs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# Import path planning module
from lsy_drone_racing.control.path_planning import PathPlanner, PathConfig

# Import drone racing framework
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crazyflow import Sim


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MPCCConfig:
    """Configuration for MPCC controller."""
    # MPC Horizon
    N_horizon: int = 40                    # Number of horizon steps
    T_horizon: float = 0.7                 # Horizon time (seconds)
    
    # Arc-length model
    model_arc_step: float = 0.05            # Arc length discretization
    model_traj_length: float = 15.0         # Trajectory length in model
    
    # Cost function weights (tunable for speed/stability trade-off)
    # Higher values = more tracking accuracy (stability)
    # Lower values = more speed
    q_lag: float = 80.0                    # Lag error weight
    q_lag_peak: float = 500.0              # Lag error weight at gates
    q_contour: float = 120.0               # Contour error weight
    q_contour_peak: float = 700.0          # Contour error weight at gates
    q_attitude: float = 1.0                # Attitude regularization
    
    # Control smoothness
    r_thrust: float = 0.2                  # Thrust rate penalty
    r_roll: float = 0.3                    # Roll rate penalty
    r_pitch: float = 0.3                   # Pitch rate penalty
    r_yaw: float = 0.50                    # Yaw rate penalty
    
    # Speed incentive
    mu_speed: float = 10.0                  # Progress reward
    w_speed_gate: float = 9.0               # Speed penalty at gates
    
    # Safety bounds
    pos_bounds: tuple = (
        (-2.6, 2.6),                        # X bounds
        (-2.0, 1.8),                        # Y bounds
        (-0.1, 2.0),                        # Z bounds
    )
    vel_bounds: tuple = (-2.0, 6.0)         # Velocity bounds (m/s)
    
    # Path planning
    planned_duration: float = 30.0          # Nominal trajectory duration
    
    # Logging settings
    log_interval: int = 100                 # Print debug info every N ticks


# =============================================================================
# MPCC Controller
# =============================================================================

class MPCCController(Controller):
    """
    Model Predictive Contouring Control for Drone Racing.
    
    This controller optimizes both tracking accuracy and progress speed
    along a pre-planned trajectory using nonlinear MPC.
    """
    
    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        mpcc_config: Optional[MPCCConfig] = None,
        path_config: Optional[PathConfig] = None
    ):
        """
        Initialize the MPCC controller.
        
        Args:
            obs: Initial observation.
            info: Initial environment info.
            config: Race configuration.
            mpcc_config: MPCC configuration. Uses defaults if None.
            path_config: Path planning configuration. Uses defaults if None.
        """
        super().__init__(obs, info, config)
        
        # Configurations
        self.mpcc_cfg = mpcc_config or MPCCConfig()
        self.path_cfg = path_config or PathConfig()
        
        # Controller state
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self.finished = False
        
        # Load dynamics parameters
        self._dyn_params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(self._dyn_params["mass"])
        self._gravity = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = self._mass * self._gravity
        
        # Initialize path planner
        self.path_planner = PathPlanner(self.path_cfg)
        
        # Store initial position
        self._initial_pos = obs["pos"].copy()
        
        # Environment change detection
        self._last_gate_flags = None
        self._last_obst_flags = None
        
        # Gate detection tracking
        num_gates = len(obs['gates_pos'])
        self._gate_detected_flags = np.zeros(num_gates, dtype=bool)
        self._gate_real_positions = np.full((num_gates, 3), np.nan)
        
        # Plan initial trajectory
        self._plan_trajectory(obs)
        
        # MPC parameters
        self.N = self.mpcc_cfg.N_horizon
        self.T = self.mpcc_cfg.T_horizon
        self.dt = self.T / self.N
        self.model_arc_step = self.mpcc_cfg.model_arc_step
        self.model_traj_length = self.mpcc_cfg.model_traj_length
        
        # State / input dimensions
        self.nx = 15  # [px,py,pz, vx,vy,vz, roll,pitch,yaw, f_collective,f_cmd, r_cmd,p_cmd,y_cmd, theta]
        self.nu = 5   # [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]

        # Build MPCC solver
        self._build_solver()
        
        # Initialize control states
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        
        # Current observation (for debug)
        self._current_pos = obs["pos"].copy()
        print(f"[MPCC] Initialized. Horizon: N={self.N}, T={self.T:.2f}s")
        print(f"[MPCC] Arc trajectory length: {self.arc_trajectory.x[-1]:.2f}")
    
    # =========================================================================
    # Trajectory Planning
    # =========================================================================
    
    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Plan or replan the trajectory."""
        print(f"[MPCC] Planning trajectory at T={self._step_count / self._ctrl_freq:.2f}s"
              if hasattr(self, '_step_count') and hasattr(self, '_ctrl_freq') else "[MPCC] Planning initial trajectory...")
    
        obs_planning = obs.copy()
        obs_planning['pos'] = self._initial_pos.copy() if hasattr(self, '_initial_pos') else obs['pos'].copy()
            
        # Use path planner to generate trajectory
        result = self.path_planner.plan_trajectory(
            obs_planning,
            trajectory_duration=self.mpcc_cfg.planned_duration,
            sampling_freq=self._ctrl_freq if hasattr(self, '_ctrl_freq') else 100,
            for_mpcc=True,
            mpcc_extension_length=self.mpcc_cfg.model_traj_length
        )
        
        # Store full result
        self._trajectory_result = result
        
        # Store results
        self.trajectory = result.spline
        self.arc_trajectory = result.arc_spline
        self.waypoints = result.waypoints
        self.total_arc_length = result.total_length
        
        # Cache for cost computation
        self._cached_gate_centers = obs["gates_pos"].copy()
        self._cached_obstacles = obs["obstacles_pos"].copy()
    
    # =========================================================================
    # MPCC Solver Construction
    # =========================================================================
    
    def _build_solver(self):
        # Build the physical model
        # NOTE: _build_dynamics_model() creates self.pd_list, self.tp_list, self.qc_dyn
        #       as MX.sym attributes directly — do NOT slice model.p afterwards.
        model = self._build_dynamics_model()
        
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        
        n_samples = int(self.model_traj_length / self.model_arc_step)  # 300
        # Parameters: 300*3 (positions) + 300*3 (tangents) + 300 (weights) = 2100
        ocp.parameter_values = np.zeros(n_samples * 3 + n_samples * 3 + n_samples)
        
        # External cost function
        ocp.cost.cost_type = "EXTERNAL"
        
        # Cost expression uses self.pd_list / self.tp_list / self.qc_dyn
        # which were created as MX.sym inside _build_dynamics_model()
        ocp.model.cost_expr_ext_cost = self._build_cost_expression()
        
        # --- INPUT CONSTRAINTS (u) ---
        # [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]
        ocp.constraints.lbu = np.array([-100.0, -100.0, -100.0, -100.0, -2.0])
        ocp.constraints.ubu = np.array([ 100.0,  100.0,  100.0,  100.0, 15.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        # --- INITIAL STATE CONSTRAINT ONLY ---
        # No intermediate state box constraints (lbx/ubx/idxbx).
        # They cause immediate QP infeasibility when the warm-start
        # violates them (e.g. f_cmd=0 at init < thrust_min).
        # Physical safety is enforced via input bounds + dynamics.
        ocp.constraints.x0 = np.zeros(self.nx)
        
        # --- SOLVER SETTINGS ---
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.T
        
        # Generate and compile C++ code
        self.solver = AcadosOcpSolver(ocp, json_file="mpcc_racing.json", verbose=False)
    
    def _build_dynamics_model(self) -> AcadosModel:
        """Build the quadrotor dynamics model with real drone parameters."""
        model_name = "mpcc_drone_racing"
        
        # Physical parameters from drone model
        mass    = self._mass
        gravity = self._gravity

        # Rate model parameters derived from drone-models
        k = np.array(self._dyn_params["rpy_coef"],       dtype=float)   # [k_roll, k_pitch, k_yaw]
        d = np.array(self._dyn_params["rpy_rates_coef"], dtype=float)   # [d_roll, d_pitch, d_yaw]
        b = np.array(self._dyn_params["cmd_rpy_coef"],   dtype=float)   # [b_roll, b_pitch, b_yaw]

        eps = 1e-9
        a    = -k / (d + eps)
        beta = -b / (d + eps)

        params_roll_rate  = [float(a[0]), float(beta[0])]
        params_pitch_rate = [float(a[1]), float(beta[1])]
        params_yaw_rate   = [float(a[2]), float(beta[2])]
        
        # --- State variables ---
        self.px    = MX.sym("px");    self.py    = MX.sym("py");    self.pz    = MX.sym("pz")
        self.vx    = MX.sym("vx");    self.vy    = MX.sym("vy");    self.vz    = MX.sym("vz")
        self.roll  = MX.sym("roll");  self.pitch = MX.sym("pitch"); self.yaw   = MX.sym("yaw")
        self.f_collective = MX.sym("f_collective")  # actual thrust (lagged)
        self.f_cmd        = MX.sym("f_cmd")         # commanded thrust
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")
        self.theta = MX.sym("theta")  # Progress along arc-length path
        
        # --- Input variables ---
        self.df_cmd     = MX.sym("df_cmd")
        self.dr_cmd     = MX.sym("dr_cmd")
        self.dp_cmd     = MX.sym("dp_cmd")
        self.dy_cmd     = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")  # Progress speed
        
        # State vector (nx = 15)
        states = vertcat(
            self.px, self.py, self.pz,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.f_collective, self.f_cmd,
            self.r_cmd, self.p_cmd, self.y_cmd,
            self.theta
        )
        # Input vector (nu = 5)
        inputs = vertcat(
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd,
            self.v_theta_cmd
        )
        
        # --- Dynamics equations ---
        thrust    = self.f_collective
        inv_mass  = 1.0 / mass
        
        ax = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * cos(self.yaw)
            + sin(self.roll) * sin(self.yaw)
        )
        ay = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * sin(self.yaw)
            - sin(self.roll) * cos(self.yaw)
        )
        az = inv_mass * thrust * cos(self.roll) * cos(self.pitch) - gravity
        
        f_dyn = vertcat(
            self.vx, self.vy, self.vz,
            ax, ay, az,
            params_roll_rate[0]  * self.roll  + params_roll_rate[1]  * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0]   * self.yaw   + params_yaw_rate[1]   * self.y_cmd,
            10.0 * (self.f_cmd - self.f_collective),   # thrust lag
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd
        )
        
        # --- Trajectory parameters (MX.sym — must be created HERE, not sliced later) ---
        n_samples = int(self.model_traj_length / self.model_arc_step)  # 300
        self.pd_list = MX.sym("pd_list", 3 * n_samples)   # reference positions
        self.tp_list = MX.sym("tp_list", 3 * n_samples)   # reference tangents
        self.qc_dyn  = MX.sym("qc_dyn",     n_samples)   # dynamic cost weights
        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn)
        
        # Build model
        model = AcadosModel()
        model.name          = model_name
        model.f_expl_expr   = f_dyn
        model.x             = states
        model.u             = inputs
        model.p             = params
        
        return model
    
    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        """CasADi-compatible piecewise linear interpolation."""
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        
        idx_low  = floor(idx_float)
        idx_high = idx_low + 1
        alpha    = idx_float - idx_low
        
        idx_low  = if_else(idx_low  < 0,   0,     idx_low)
        idx_high = if_else(idx_high >= M, M - 1, idx_high)
        
        p_low  = vertcat(*[flattened_points[dim * idx_low  + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        
        return (1.0 - alpha) * p_low + alpha * p_high
    
    def _encode_trajectory_params(self) -> np.ndarray:
        """Encode current trajectory into the parameter vector for acados."""
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        
        pd_vals = self.arc_trajectory(theta_samples)
        tp_vals = self.arc_trajectory.derivative(1)(theta_samples)
        
        # Dynamic cost weights: higher near gates and obstacles
        qc_dyn = np.zeros_like(theta_samples)
        
        for gate_center in self._cached_gate_centers:
            d_gate  = np.linalg.norm(pd_vals - gate_center, axis=-1)
            qc_gate = 0.4 * np.exp(-8.0 * d_gate**2)
            qc_dyn  = np.maximum(qc_dyn, qc_gate)
        
        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            qc_obs   = 0.2 * np.exp(-8.0 * d_obs_xy**2)
            qc_dyn   = np.maximum(qc_dyn, qc_obs)
        
        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_dyn])
    
    def _build_cost_expression(self):
        """Build MPCC stage cost expression in CasADi symbolic form."""
        cfg = self.mpcc_cfg
        
        position = vertcat(self.px, self.py, self.pz)
        attitude = vertcat(self.roll, self.pitch, self.yaw)
        control  = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        
        # Interpolate reference position, tangent and weight at current theta
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_dyn, dim=1)
        
        # Contouring / lag errors
        tp_unit   = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta   = position - pd_theta
        e_lag     = dot(tp_unit, e_theta) * tp_unit   # along path
        e_contour = e_theta - e_lag                   # perpendicular to path
        
        # Tracking cost (with dynamic gate/obstacle weights)
        Q_w        = cfg.q_attitude * DM(np.eye(3))
        track_cost = (
            (cfg.q_lag     + cfg.q_lag_peak     * qc_theta) * dot(e_lag,     e_lag)
            + (cfg.q_contour + cfg.q_contour_peak * qc_theta) * dot(e_contour, e_contour)
            + attitude.T @ Q_w @ attitude
        )
        
        # Control smoothness cost
        R_df        = DM(np.diag([cfg.r_thrust, cfg.r_roll, cfg.r_pitch, cfg.r_yaw]))
        smooth_cost = control.T @ R_df @ control
        
        # Speed incentive (maximize progress, penalize speed near gates)
        speed_cost = -cfg.mu_speed * self.v_theta_cmd + cfg.w_speed_gate * qc_theta * (self.v_theta_cmd ** 2)
        
        return track_cost + smooth_cost + speed_cost
    
    # =========================================================================
    # Environment Change Detection
    # =========================================================================
    
    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect changes in gate/obstacle visited flags."""
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False
        
        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst  = np.array(obs.get("obstacles_visited", []), dtype=bool)
        
        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False
        if curr_obst.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr_obst
            return False
        
        gate_trigger = np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = np.any((~self._last_obst_flags) & curr_obst)
        
        # Record newly detected gate positions
        for i, is_visited in enumerate(curr_gates):
            if is_visited and not self._gate_detected_flags[i]:
                self._gate_detected_flags[i] = True
                self._gate_real_positions[i]  = obs['gates_pos'][i]
                print(f"[GATE DETECTED] Gate {i+1} at real position: "
                      f"[{obs['gates_pos'][i][0]:.3f}, {obs['gates_pos'][i][1]:.3f}, {obs['gates_pos'][i][2]:.3f}]")
        
        self._last_gate_flags = curr_gates.copy()
        self._last_obst_flags = curr_obst.copy()
        
        return bool(gate_trigger or obst_trigger)
    
    # =========================================================================
    # Safety Checks
    # =========================================================================
    
    def _check_position_bounds(self, pos: NDArray[np.floating]) -> bool:
        """Check if position is within safe bounds."""
        for i, (low, high) in enumerate(self.mpcc_cfg.pos_bounds):
            if pos[i] < low or pos[i] > high:
                return False
        return True
    
    def _check_velocity_bounds(self, vel: NDArray[np.floating]) -> bool:
        """Check if speed is within safe range. Low-speed OK (hover/start)."""
        speed = np.linalg.norm(vel)
        _, high = self.mpcc_cfg.vel_bounds
        return speed < high  # only check upper bound — zero speed is fine at start
    
    # =========================================================================
    # Main Control Loop
    # =========================================================================
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Compute control command using MPCC.
        
        Returns:
            Control command [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd].
        """
        self._current_pos = obs["pos"].copy()
        
        # --- Replan if environment changed ---
        if self._detect_environment_change(obs):
            print("[MPCC] Environment change detected, replanning...")
            self._plan_trajectory(obs)
            try:
                theta_proj, _ = self.path_planner.find_closest_point(
                    self.arc_trajectory, obs["pos"]
                )
                self.last_theta = max(self.last_theta, float(theta_proj))
            except Exception as e:
                print(f"[MPCC] Warning: could not project theta after replanning: {e}")
            
            # Push updated parameters to every horizon stage
            param_vec = self._encode_trajectory_params()
            for k in range(self.N + 1):
                self.solver.set(k, "p", param_vec)
        
        # --- Encode trajectory parameters every step ---
        param_vec = self._encode_trajectory_params()
        for k in range(self.N + 1):
            self.solver.set(k, "p", param_vec)
        
        # --- Convert quaternion → Euler ---
        roll, pitch, yaw = Rotation.from_quat(obs["quat"]).as_euler("xyz")
        
        # --- Build current state vector (nx = 15) ---
        # [px,py,pz, vx,vy,vz, roll,pitch,yaw, f_collective(9),f_cmd(10), r_cmd(11),p_cmd(12),y_cmd(13), theta(14)]
        x_now = np.concatenate([
            obs["pos"],
            obs["vel"],
            np.array([roll, pitch, yaw]),
            np.array([self.last_f_collective, self.last_f_cmd]),
            self.last_rpy_cmd,
            np.array([self.last_theta])
        ])

        # --- Fix initial state FIRST (before warm start injection) ---
        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)
        
        # --- Warm start ---
        if not hasattr(self, "_x_warm"):
            # First call: replicate current state across entire horizon.
            # x_now already has hover_thrust in f_collective(9) and f_cmd(10),
            # so the warm-start is physically consistent from the start.
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(self.nu) for _ in range(self.N)]
            # Seed theta progression along horizon
            for i in range(self.N + 1):
                self._x_warm[i][14] = float(self.last_theta) + i * (self.dt * 1.0)
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]

        # Always pin stage 0 to true current state
        self._x_warm[0] = x_now.copy()

        for i in range(self.N):
            self.solver.set(i, "x", self._x_warm[i])
            self.solver.set(i, "u", self._u_warm[i])
        self.solver.set(self.N, "x", self._x_warm[self.N])
        
        # --- Termination checks ---
        if self.last_theta >= float(self.arc_trajectory.x[-1]):
            self.finished = True
            print("[MPCC] Finished: reached end of path")
        
        if not self._check_position_bounds(obs["pos"]):
            self.finished = True
            print("[MPCC] Finished: position out of bounds")
        
        if not self._check_velocity_bounds(obs["vel"]):
            self.finished = True
            print("[MPCC] Finished: velocity out of bounds")
        
        # --- Solve ---
        status = self.solver.solve()
        if status != 0:
            print(f"[MPCC] Solver returned status {status}")
            # On failure: keep last known good commands, reset warm start
            if hasattr(self, "_x_warm"):
                del self._x_warm
            if hasattr(self, "_u_warm"):
                del self._u_warm
            cmd = np.array([
                self.last_rpy_cmd[0],
                self.last_rpy_cmd[1],
                self.last_rpy_cmd[2],
                self.last_f_cmd
            ], dtype=np.float32)
            self._step_count += 1
            return cmd
        
        # --- Extract solution ---
        self._x_warm = [self.solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.solver.get(i, "u") for i in range(self.N)]
        
        # x = [px,py,pz, vx,vy,vz, roll,pitch,yaw, f_collective(9), f_cmd(10), r_cmd(11),p_cmd(12),y_cmd(13), theta(14)]
        x_next = self.solver.get(1, "x")
        
        self.last_f_collective = float(x_next[9])
        self.last_f_cmd        = float(x_next[10])
        self.last_rpy_cmd      = np.array(x_next[11:14])
        self.last_theta        = float(x_next[14])
        
        cmd = np.array([
            self.last_rpy_cmd[0],
            self.last_rpy_cmd[1],
            self.last_rpy_cmd[2],
            self.last_f_cmd
        ], dtype=np.float32)
        
        # --- Periodic logging ---
        if self._step_count % self.mpcc_cfg.log_interval == 0:
            print(f"[MPCC] T={self._step_count / self._ctrl_freq:.2f}s, "
                  f"theta={self.last_theta:.2f}/{self.arc_trajectory.x[-1]:.2f}, "
                  f"cmd=[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}, {cmd[3]:.2f}]")
        
        self._step_count += 1
        return cmd
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        """Called after each environment step."""
        return self.finished
    
    def episode_callback(self):
        """Called at episode reset."""
        print("[MPCC] Episode reset")
        self._step_count = 0
        self.finished = False
        
        for attr in ["_last_gate_flags", "_last_obst_flags", "_x_warm", "_u_warm"]:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.last_theta        = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd        = self.hover_thrust
        self.last_rpy_cmd      = np.zeros(3)

    def render_callback(self, sim: "Sim"):
        """Draw the reference trajectory in green in the simulator."""
        if getattr(self, "arc_trajectory", None) is None:
            return
        thetas   = np.linspace(0.0, float(self.arc_trajectory.x[-1]), 200)
        ref_path = np.asarray(self.arc_trajectory(thetas), dtype=float)
        draw_line(sim, ref_path, rgba=(0.0, 1.0, 0.0, 1.0))

    # =========================================================================
    # Debug / Utility
    # =========================================================================
    
    def get_debug_lines(self):
        """Return line segments for external visualization."""
        debug_lines = []
        
        if hasattr(self, "arc_trajectory"):
            try:
                full_path = self.arc_trajectory(self.arc_trajectory.x)
                debug_lines.append((full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
            except Exception:
                pass
        
        if hasattr(self, "_x_warm"):
            try:
                pred_states = np.array([x[:3] for x in self._x_warm])
                debug_lines.append((pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
            except Exception:
                pass
        
        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory"):
            try:
                target  = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_pos, target])
                debug_lines.append((segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0))
            except Exception:
                pass
        
        return debug_lines
    
    def get_trajectory(self) -> CubicSpline:
        """Get the current time-parameterized trajectory spline."""
        return self.trajectory
    
    def get_arc_trajectory(self) -> CubicSpline:
        """Get the arc-length parameterized trajectory spline."""
        return self.arc_trajectory
    
    def get_progress(self) -> float:
        """Get current progress ratio (0 to 1)."""
        if hasattr(self, "arc_trajectory"):
            return self.last_theta / self.arc_trajectory.x[-1]
        return 0.0