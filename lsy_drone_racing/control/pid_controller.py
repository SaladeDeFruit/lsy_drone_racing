from __future__ import annotations

import os    # <-- NOUVEAU IMPORT POUR LE CHEMIN DU FICHIER
import time  # <-- IMPORT AJOUTÉ POUR LE CHRONOMÈTRE
from typing import TYPE_CHECKING

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.combinations_path_generator import (
    chord_length_param,
    optimal_waypoints,
)

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
        # Demander au générateur de combinaisons les waypoints du chemin le plus rapide :
        # il classe tous les (ordre de gates x faces) par durée TOPP-RA et renvoie les
        # waypoints (start + approche/centre/sortie par gate) de la combinaison optimale.
        waypoints = optimal_waypoints(start_pos, obs["gates_pos"], obs["gates_quat"])

        # 1. Create a geometric path from the waypoints. Use distance-proportional knots
        # (chord length), NOT np.linspace: with uniform knots the cubic spline overshoots and
        # loops at each gate (the close approach/center/exit triplet vs the ~1 m gate jumps).
        ss = chord_length_param(waypoints)
        path = ta.SplineInterpolator(ss, waypoints)

        # 2. Define kinematic constraints for the drone
        v_max_xy = 1.8
        v_max_z = 1
        vbounds = np.array([
            [-v_max_xy, v_max_xy],
            [-v_max_xy, v_max_xy],
            [-v_max_z, v_max_z]
        ])
        
        a_max_xy = 2.5
        abounds = np.array([
            [-a_max_xy, a_max_xy],
            [-a_max_xy, a_max_xy],
            [-5, 3]
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