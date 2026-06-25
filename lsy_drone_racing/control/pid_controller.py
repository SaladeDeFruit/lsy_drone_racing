from __future__ import annotations

import os    # <-- NOUVEAU IMPORT POUR LE CHEMIN DU FICHIER
import time  # <-- IMPORT AJOUTÉ POUR LE CHRONOMÈTRE
from typing import TYPE_CHECKING

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.combinations_path_generator import (
    chord_length_param,
    rank_combinations_by_duration,
)
from lsy_drone_racing.control.TOGTPOptimizer import (
    DroneRacingOptimizer,
    PolygonGate,
    QuadrotorModel,
)

GATE_OPENING = 0.4  # m, square gate inner opening (see config/level0.toml)


def togt_optimized_waypoints(start_pos, gates_pos, gates_quat, exit_dist=0.3, gate_use_frac=0.1):
    """Run the TOGT optimizer (DroneRacingOptimizer) and return its end-of-pipeline waypoints.

    The gates are first reordered into the SEQUENCE chosen by ``combinations_path_generator``
    (the fastest gate order by TOPP-RA ranking), then handed to the optimizer in that sequence.
    The optimizer optimizes, per gate, WHERE inside the gate opening the path crosses, plus the
    per-segment times, minimizing total time under a motor-thrust penalty. We then rebuild the
    geometric waypoints from its solution:
    ``[start, crossing_gate_0, ..., crossing_gate_n, end]``. These feed TOPP-RA downstream.

    Args:
        start_pos: (3,) drone start position (``obs["pos"]``).
        gates_pos: (n_gates, 3) gate centers (``obs["gates_pos"]``).
        gates_quat: (n_gates, 4) xyzw gate orientations (``obs["gates_quat"]``).
        exit_dist: how far past the last gate (along its normal) to place the end point, in meters.
        gate_use_frac: fraction of the half-opening the crossing may use. The optimizer pushes the
            crossing to a polygon CORNER (worst case in-plane offset = half * sqrt(2)); shrinking
            the polygon keeps that corner safely inside the real opening (and clear of the frame).

    Returns:
        (n_gates + 2, 3) array of waypoints in traversal order.
    """
    gates_pos = np.asarray(gates_pos, dtype=float)
    gates_quat = np.asarray(gates_quat, dtype=float)
    start_pos = np.asarray(start_pos, dtype=float)

    # Reorder the gates into the sequence given by combinations_path_generator (fastest order by
    # TOPP-RA ranking), so the TOGT optimizer receives the gates in that traversal order.
    order = rank_combinations_by_duration(start_pos, gates_pos, gates_quat)[0][0]
    order = list(order)
    gates_pos = gates_pos[order]
    gates_quat = gates_quat[order]

    half = (GATE_OPENING / 2.0) * gate_use_frac  # safety margin: corner stays inside the opening

    # Build a PolygonGate (4 corners of the opening) per gate, in the gate plane (y, z axes).
    gates = []
    for g in range(len(gates_pos)):
        rot = R.from_quat(gates_quat[g])
        y_axis, z_axis = rot.apply([0.0, 1.0, 0.0]), rot.apply([0.0, 0.0, 1.0])
        c = gates_pos[g]
        verts = [
            c + half * y_axis + half * z_axis,
            c - half * y_axis + half * z_axis,
            c - half * y_axis - half * z_axis,
            c + half * y_axis - half * z_axis,
        ]
        gates.append(PolygonGate(g, verts))

    # End point: just past the last gate, on the side the drone is travelling toward.
    last_normal = R.from_quat(gates_quat[-1]).apply([1.0, 0.0, 0.0])
    incoming = gates_pos[-1] - (gates_pos[-2] if len(gates_pos) > 1 else start_pos)
    sign = 1.0 if np.dot(incoming, last_normal) >= 0 else -1.0
    end_pos = gates_pos[-1] + sign * exit_dist * last_normal

    result = DroneRacingOptimizer(QuadrotorModel(), gates, start_pos, end_pos).solve()
    d_vars = result.x[: len(gates) * 4].reshape((len(gates), 4))

    waypoints = [start_pos]
    for i, gate in enumerate(gates):
        waypoints.append(gate.smooth_surjection(d_vars[i]))
    waypoints.append(end_pos)
    return np.array(waypoints)

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
        # Récupérer la trajectoire optimisée en fin de pipeline par le TOGT optimizer
        # (DroneRacingOptimizer) : gates dans l'ordre de la course, point de passage optimal
        # dans chaque porte + temps optimaux. On en extrait les waypoints géométriques.
        waypoints = togt_optimized_waypoints(start_pos, obs["gates_pos"], obs["gates_quat"])

        # 1. Create a geometric path from the waypoints. Use distance-proportional knots
        # (chord length), NOT np.linspace: with uniform knots the cubic spline overshoots and
        # loops at each gate (the close approach/center/exit triplet vs the ~1 m gate jumps).
        ss = chord_length_param(waypoints)
        path = ta.SplineInterpolator(ss, waypoints)

        # 2. Define kinematic constraints for the drone
        v_max_xy = 1
        v_max_z = 1
        vbounds = np.array([
            [-v_max_xy, v_max_xy],
            [-v_max_xy, v_max_xy],
            [-v_max_z, v_max_z]
        ])
        
        a_max_xy = 1
        abounds = np.array([
            [-a_max_xy, a_max_xy],
            [-a_max_xy, a_max_xy],
            [-1, 1]
        ])
        
        pc_vel = constraint.JointVelocityConstraint(vbounds)
        pc_acc = constraint.JointAccelerationConstraint(abounds)

        # 3. Setup and solve the TOPP-RA problem
        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
        self._trajectory = instance.compute_trajectory()

        if self._trajectory is None:
            raise RuntimeError("TOPP-RA failed to compute a valid trajectory. Check your waypoints and constraints.")

        self._t_total = self._trajectory.duration
        print(f"Computed TOPP-RA trajectory with optimal duration: {self._t_total:.2f} s")

        self._tick = 0
        self._finished = False
        
        # --- VARIABLES POUR LE TEMPS RÉEL ---
        self._real_start_time = None
        self._time_logged = False
        
        # --- NOUVEAU : PRÉPARATION DU FICHIER ET PREMIÈRE ÉCRITURE ---
        # On définit le chemin absolu vers le dossier actuel
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._log_file = os.path.join(current_dir, "temps_reel_trajet.txt")
        
        # On écrit la première partie de la ligne (SANS retour à la ligne '\n')
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"Temps simulé: {self._t_total:.4f} s | Temps réel: ")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        
        # --- DÉMARRER LE CHRONO AU PREMIER MOUVEMENT ---
        if self._real_start_time is None:
            self._real_start_time = time.perf_counter()

        t = min(self._tick / self._freq, self._t_total)
        
        # --- VÉRIFIER LA FIN ET COMPLÉTER LE FICHIER ---
        if t >= self._t_total and not self._finished: 
            self._finished = True
            
            if not self._time_logged:
                # Calcul de la durée d'exécution réelle
                real_duration = time.perf_counter() - self._real_start_time
                
                # On complète la ligne existante et on ajoute le retour à la ligne (\n) à la fin
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"{real_duration:.4f} s\n")
                
                self._time_logged = True

        des_pos = self._trajectory(t, 0)
        des_vel = self._trajectory(t, 1)
        des_acc = self._trajectory(t, 2)

        yaw_and_rates = np.zeros(4) 
        action = np.concatenate((des_pos, des_vel, des_acc,(0,0.1,0,0)), dtype=np.float32)
        
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
        # --- SÉCURITÉ : Si l'épisode précédent a été coupé avant d'atteindre la fin ---
        if self._real_start_time is not None and not self._time_logged:
             with open(self._log_file, "a", encoding="utf-8") as f:
                 f.write("Inachevé (Crash ou Reset)\n")

        self._tick = 0
        self._real_start_time = None
        self._time_logged = False
        self._finished = False

        # --- NOUVEAU : On réécrit le début de la ligne pour le nouvel épisode ---
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"Temps simulé: {self._t_total:.4f} s | Temps réel: ")

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        current_time = min(self._tick / self._freq, self._t_total)
        
        setpoint = self._trajectory(current_time, 0).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        
        t_samples = np.linspace(0, self._t_total, 100)
        trajectory_points = np.array([self._trajectory(t, 0) for t in t_samples])
        draw_line(sim, trajectory_points, rgba=(0.0, 1.0, 0.0, 1.0))