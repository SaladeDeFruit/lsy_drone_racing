from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import make_interp_spline
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

GATE_OPENING = 0.4
lookat = [0.0, 0.0, 0.0]
  # m, square gate inner opening
SHOW_MATPLOTLIB_DEBUG = False  # METTRE À FALSE POUR NE PAS BLOQUER LA SIMULATION

# Le drone se stabilise quelques cm AU-DESSUS de la consigne (erreur de suivi statique en z).
# On commande donc une altitude un peu plus basse pour compenser. Augmenter si le drone reste
# trop haut, diminuer s'il vole trop bas.
Z_OFFSET = 0.05  # m, abaissement de la consigne d'altitude


def plot_debug_3d(gates_verts, waypoints, start_pos, end_pos):
    """Affiche la trajectoire finale et les portes dans Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les portes (vert)
    for verts in gates_verts:
        ax.add_collection3d(Poly3DCollection([verts], alpha=0.15, facecolor='green', edgecolor='black'))

    # Tracer la trajectoire optimisée
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'b.-', linewidth=2, label='Trajectoire TOGT')
    ax.scatter(*start_pos, color='k', s=50, label='Départ')
    ax.scatter(*end_pos, color='m', s=50, label='Arrivée')

    # Configuration de la vue
    all_pts = np.vstack([waypoints, start_pos, end_pos])
    ax.set_xlim([all_pts[:, 0].min()-1, all_pts[:, 0].max()+1])
    ax.set_ylim([all_pts[:, 1].min()-1, all_pts[:, 1].max()+1])
    ax.set_zlim([all_pts[:, 2].min()-1, all_pts[:, 2].max()+1])
    ax.set_title("Optimisation TOGT")
    ax.legend()
    plt.show()


def add_lead_in_out(waypoints, gates_quat, lead=0.35):
    out = [waypoints[0]]
    prev = waypoints[0]
    for i in range(len(gates_quat)):
        apex = waypoints[1 + i]
        n = R.from_quat(gates_quat[i]).apply([1.0, 0.0, 0.0])
        if np.dot(n, apex - prev) < 0:
            n = -n
        out.append(apex - lead * n)
        out.append(apex)
        out.append(apex + lead * n)
        prev = apex
    out.append(waypoints[-1])
    return np.array(out)


def togt_optimized_waypoints(start_pos, gates_pos, gates_quat, exit_dist=0.3, gate_use_frac=0.9,
                             reorder=True, return_order=False):
    gates_pos = np.asarray(gates_pos, dtype=float)
    gates_quat = np.asarray(gates_quat, dtype=float)
    start_pos = np.asarray(start_pos, dtype=float)

    # Réorganisation des portes par durée (ordre TOGT le plus rapide). À désactiver
    # (reorder=False) quand la course impose l'ordre des indices (cf. obs["target_gate"]).
    if reorder:
        order = list(rank_combinations_by_duration(start_pos, gates_pos, gates_quat)[0][0])
    else:
        order = list(range(len(gates_pos)))
    gates_pos = gates_pos[order]
    gates_quat = gates_quat[order]

    half = (GATE_OPENING / 2.0) * gate_use_frac

    # Extraire les coins des portes
    gates_verts = []
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
        gates_verts.append(verts)

    # Création des objets PolygonGate pour le TOGT
    gates = []
    for g, verts in enumerate(gates_verts):
        gates.append(PolygonGate(g, verts))

    # Point de fin
    last_normal = R.from_quat(gates_quat[-1]).apply([1.0, 0.0, 0.0])
    incoming = gates_pos[-1] - (gates_pos[-2] if len(gates_pos) > 1 else start_pos)
    sign = 1.0 if np.dot(incoming, last_normal) >= 0 else -1.0
    end_pos = gates_pos[-1] + sign * exit_dist * last_normal

    # Résolution TOGT
    inst = DroneRacingOptimizer(QuadrotorModel(), gates, start_pos, end_pos)
    result = inst.solve()

    # Reconstruire les waypoints géométriques
    d_vars = result.x[: len(gates) * 4].reshape((len(gates), 4))
    k_vars = result.x[len(gates) * 4:]
    t_segments = inst.transform_time(k_vars)
    waypoints, _ = inst._build_waypoints(d_vars, t_segments)
    waypoints_array = np.array(waypoints)
    waypoints_array = add_lead_in_out(waypoints_array, gates_quat, lead=0.02)

    if SHOW_MATPLOTLIB_DEBUG:
        plot_debug_3d(gates_verts, waypoints_array, start_pos, end_pos)

    if return_order:
        return waypoints_array, order
    return waypoints_array


if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class QuinticTrajectory:
    """Trajectoire spline quintique (degré 5), avec la même interface que la trajectoire
    TOPP-RA : ``traj(t, order)`` (order 0/1/2 = pos/vit/acc) et ``.duration``.

    Construite pour imposer exactement (position, vitesse, accélération) au départ — le point
    recalculé — donc continuité C2 : aucune discontinuité de vitesse/accélération lors d'une
    re-planification en vol.
    """

    def __init__(self, spline, duration):
        self._spline = spline
        self._d1 = spline.derivative(1)
        self._d2 = spline.derivative(2)
        self.duration = float(duration)

    def __call__(self, t, order: int = 0):
        t = float(np.clip(t, 0.0, self.duration))
        if order == 0:
            return np.asarray(self._spline(t), dtype=float)
        if order == 1:
            return np.asarray(self._d1(t), dtype=float)
        return np.asarray(self._d2(t), dtype=float)


class StateController(Controller):
    """State controller following a pre-defined TOPP-RA trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        ta.setup_logging("WARNING")
        start_pos = np.asarray(obs["pos"], dtype=float)

        # Bornes cinématiques pour TOPP-RA, réutilisées à chaque (re-)planification.
        v_max_xy = 1
        v_max_z = 1
        self._vbounds = np.array([
            [-v_max_xy, v_max_xy],
            [-v_max_xy, v_max_xy],
            [-v_max_z, v_max_z]
        ])
        a_max_xy = 1
        self._abounds = np.array([
            [-a_max_xy, a_max_xy],
            [-a_max_xy, a_max_xy],
            [-1, 1]
        ])

        self._tick = 0
        self._finished = False
        self._real_start_time = None
        self._time_logged = False

        # On garde l'ordre TOGT (le plus rapide) pour voler. self._order[j] = indice config de
        # la j-ème porte balayée ; self._waypoints_full est donc dans cet ordre de traversée.
        self._waypoints_full, self._order = togt_optimized_waypoints(
            start_pos, obs["gates_pos"], obs["gates_quat"], return_order=True
        )

        # Indice (dans l'ordre de balayage) de la prochaine porte à corriger dès qu'elle se révèle.
        self._next_j = 0

        # Trajectoire initiale : on suit le plan nominal jusqu'à la première découverte.
        self._build_trajectory(self._waypoints_full)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._log_file = os.path.join(current_dir, "temps_reel_trajet.txt")

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"Temps simulé: {self._t_total:.4f} s | Temps réel: ")

    def _build_trajectory(self, waypoints):
        """Construit (ou reconstruit) la trajectoire TOPP-RA à partir d'une liste de waypoints.

        cubic spline d'interpolation (``ta.SplineInterpolator``) + re-temporisation TOPP-RA.
        Remet l'horloge à zéro : la nouvelle trajectoire repart de son ``t = 0`` (premier
        waypoint). Appelée au démarrage (plan nominal) puis à chaque découverte d'une porte.
        """
        waypoints = np.asarray(waypoints, dtype=float)
        self._waypoints = waypoints

        ss = chord_length_param(waypoints)
        path = ta.SplineInterpolator(ss, waypoints)

        pc_vel = constraint.JointVelocityConstraint(self._vbounds)
        pc_acc = constraint.JointAccelerationConstraint(self._abounds)
        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
        trajectory = instance.compute_trajectory()

        if trajectory is None:
            raise RuntimeError("TOPP-RA failed to compute a valid trajectory. Check your waypoints and constraints.")

        self._trajectory = trajectory
        self._t_total = trajectory.duration
        self._tick = 0  # la nouvelle trajectoire repart de son début
        print(f"[plan] trajectoire (re)calculée : {len(waypoints)} waypoints, durée {self._t_total:.2f} s")

    def _build_corrected_trajectory(self, waypoints, v0, a0, n_samples: int = 60):
        """Re-planification CONTINUE (C2) : injecte l'état courant (v0, a0) au point recalculé.

        1) TOPP-RA fournit le timing time-optimal et respecte les contraintes sur la géométrie
           (cubic spline d'interpolation des waypoints).
        2) On ré-ajuste une spline QUINTIQUE (ordre 5) sur des échantillons temporels de cette
           trajectoire, en imposant au départ (vitesse = v0, accélération = a0) et en fin les
           (vitesse, accélération) de TOPP-RA (≈ arrêt). Résultat : profil quasi time-optimal
           mais sans à-coup de vitesse/accélération au moment de la re-planification.
        """
        waypoints = np.asarray(waypoints, dtype=float)
        self._waypoints = waypoints

        ss = chord_length_param(waypoints)
        path = ta.SplineInterpolator(ss, waypoints)
        pc_vel = constraint.JointVelocityConstraint(self._vbounds)
        pc_acc = constraint.JointAccelerationConstraint(self._abounds)
        toppra_traj = algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        ).compute_trajectory()
        if toppra_traj is None:
            raise RuntimeError("TOPP-RA a échoué pendant la re-planification corrigée.")

        T = float(toppra_traj.duration)
        ts = np.linspace(0.0, T, n_samples)
        pos = np.array([toppra_traj(t, 0) for t in ts])          # (n_samples, 3)
        v_end = np.asarray(toppra_traj(T, 1), dtype=float)
        a_end = np.asarray(toppra_traj(T, 2), dtype=float)

        # bc_type ordre 5 : 2 conditions à chaque bord (1re et 2e dérivées).
        bc = (
            [(1, np.asarray(v0, dtype=float)), (2, np.asarray(a0, dtype=float))],
            [(1, v_end), (2, a_end)],
        )
        quintic = make_interp_spline(ts, pos, k=5, bc_type=bc)

        self._trajectory = QuinticTrajectory(quintic, T)
        self._t_total = T
        self._tick = 0
        print(f"[plan] correction quintique (ordre 5) : durée {T:.2f} s, v0={np.round(v0, 2)}")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        # --- Correction locale : on corrige chaque porte dès qu'elle se révèle, DANS L'ORDRE
        # où le drone les balaie (ordre TOGT). self._order[self._next_j] = indice config de la
        # prochaine porte balayée ; on attend que sa vraie position soit révélée (gates_visited).
        visited = np.asarray(obs["gates_visited"]).reshape(-1)
        if self._next_j < len(self._order) and visited[self._order[self._next_j]]:
            j = self._next_j
            g = self._order[j]                                              # porte balayée (indice config)
            self._next_j += 1
            discovery_pos = np.asarray(obs["pos"], dtype=float)
            true_center = np.asarray(obs["gates_pos"], dtype=float)[g]       # vraie position révélée
            # Rejoint le nominal LE PLUS TÔT possible : après le vrai centre, on reprend la
            # trajectoire nominale dès l'apex de la porte (indice 2 + 3*j) au lieu d'attendre le
            # lead-out (3 + 3*j). La partie adaptée se limite à [position courante -> vrai centre].
            rejoin = self._waypoints_full[2 + 3 * j:]                        # apex_j, lead_out_j, ... -> fin
            new_wps = np.vstack([discovery_pos, true_center, rejoin])

            # État courant injecté pour une re-planification continue (C2, sans à-coup) :
            # v0 = vitesse réelle mesurée, a0 = accélération qu'on commandait juste avant.
            t_now = min(self._tick / self._freq, self._t_total)
            v0 = np.asarray(obs["vel"], dtype=float)
            a0 = np.asarray(self._trajectory(t_now, 2), dtype=float)
            try:
                self._build_corrected_trajectory(new_wps, v0, a0)
            except Exception as e:  # robustesse : repli sur un plan depuis l'arrêt
                print(f"[plan] correction quintique échouée ({e!r}) ; repli TOPP-RA depuis l'arrêt.")
                self._build_trajectory(new_wps)

        if self._real_start_time is None:
            self._real_start_time = time.perf_counter()

        t = min(self._tick / self._freq, self._t_total)

        if t >= self._t_total and not self._finished:
            self._finished = True

            if not self._time_logged:
                real_duration = time.perf_counter() - self._real_start_time
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"{real_duration:.4f} s\n")
                self._time_logged = True

        des_pos = self._trajectory(t, 0).copy()
        des_pos[2] -= Z_OFFSET  # abaisse la consigne d'altitude pour compenser le suivi statique en z
        des_vel = self._trajectory(t, 1)
        des_acc = self._trajectory(t, 2)

        yaw_and_rates = np.zeros(4)
        action = np.concatenate((des_pos, des_vel, des_acc, (0, 0.1, 0, 0)), dtype=np.float32)

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
        self._tick += 1
        return self._finished

    def episode_callback(self):
        if self._real_start_time is not None and not self._time_logged:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write("Inachevé (Crash ou Reset)\n")

        self._tick = 0
        self._real_start_time = None
        self._time_logged = False
        self._finished = False

        # Restaure le plan nominal complet pour le nouvel épisode (les corrections locales
        # seront ré-appliquées au fil des découvertes).
        self._next_j = 0
        self._build_trajectory(self._waypoints_full)

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"Temps simulé: {self._t_total:.4f} s | Temps réel: ")

    def render_callback(self, sim: Sim):
        current_time = min(self._tick / self._freq, self._t_total)

        setpoint = self._trajectory(current_time, 0).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        t_samples = np.linspace(0, self._t_total, 100)
        trajectory_points = np.array([self._trajectory(t, 0) for t in t_samples])
        draw_line(sim, trajectory_points, rgba=(0.0, 1.0, 0.0, 1.0))