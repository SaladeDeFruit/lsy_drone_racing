"""Génération de trajectoire passant par chaque gate.

Ce module construit une trajectoire lisse (cubic spline) qui traverse toutes
les portes du circuit. Pour chaque gate, on génère trois points alignés sur la
normale de la porte :

    face d'entrée  ->  centre  ->  face de sortie

Ces trois points sont colinéaires (le long de la normale de la gate), ce qui
force la spline à traverser la porte bien droite, perpendiculairement aux deux
faces : c'est le "centre tangent". On obtient donc 3 points par gate.

Pour l'instant on ne s'occupe que des gates (les obstacles sont ignorés).
Inspiré de ``path_planning.py``, réduit au cas "gates uniquement".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gate_normals(gates_quat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extrait la normale (direction de traversée) de chaque gate.

    La normale est le premier axe (x) du repère local de la porte, c'est-à-dire
    la première colonne de sa matrice de rotation.

    Args:
        gates_quat: Orientations des gates en quaternions [x, y, z, w], shape (N, 4).

    Returns:
        Vecteurs normaux des gates, shape (N, 3).
    """
    rotation_matrices = Rotation.from_quat(gates_quat).as_matrix()
    return rotation_matrices[:, :, 0]


class PathGenerator:
    """Génère une trajectoire cubic spline passant par chaque face de chaque gate."""

    def __init__(self, gate_offset: float = 0.3, duration: float = 18.0):
        """Initialise le générateur de trajectoire.

        Args:
            gate_offset: Distance (m) du centre de la gate aux points de
                face d'entrée / de sortie le long de la normale.
            duration: Durée totale (s) à laquelle la trajectoire est
                paramétrée.
        """
        self.gate_offset = gate_offset
        self.duration = duration

    def gate_waypoints(
        self,
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        reference_pos: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Génère 3 waypoints alignés sur la normale pour chaque gate.

        Pour chaque gate i on produit, dans l'ordre de traversée :
        ``centre - offset*normale`` (face d'entrée), ``centre``, puis
        ``centre + offset*normale`` (face de sortie).

        La normale est orientée selon le sens de progression (depuis le point
        de référence précédent) afin que la face d'entrée soit bien atteinte en
        premier et éviter que la spline ne fasse demi-tour.

        Args:
            gates_pos: Positions des centres des gates, shape (N, 3).
            gates_quat: Quaternions des gates [x, y, z, w], shape (N, 4).
            reference_pos: Point de départ (ex: position du drone) servant à
                orienter la première normale. Si None, on utilise le centre de
                la première gate.

        Returns:
            Waypoints ordonnés, shape (3 * N, 3).
        """
        gates_pos = np.asarray(gates_pos, dtype=float)
        normals = gate_normals(gates_quat)

        # Oriente chaque normale dans le sens de progression du circuit.
        prev = np.asarray(reference_pos, dtype=float) if reference_pos is not None else gates_pos[0]
        oriented = np.empty_like(normals)
        for i, (center, normal) in enumerate(zip(gates_pos, normals)):
            travel = center - prev
            if np.dot(travel, normal) < 0:
                normal = -normal
            oriented[i] = normal
            prev = center

        # 3 points colinéaires par gate : entrée, centre, sortie.
        entry = gates_pos - self.gate_offset * oriented
        exit_ = gates_pos + self.gate_offset * oriented

        # Entrelace en (entry_0, center_0, exit_0, entry_1, ...).
        waypoints = np.stack([entry, gates_pos, exit_], axis=1).reshape(-1, 3)
        return waypoints

    def create_spline(self, waypoints: NDArray[np.floating]) -> CubicSpline:
        """Construit une cubic spline passant par les waypoints.

        La paramétrisation temporelle suit la longueur d'arc cumulée (distance
        cordale), ce qui donne une vitesse plus uniforme le long du chemin.

        Args:
            waypoints: Points 3D à interpoler, shape (M, 3).

        Returns:
            Spline cubique évaluable en fonction du temps [0, duration].
        """
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        time_parameters = cumulative / (cumulative[-1] + 1e-6) * self.duration
        return CubicSpline(time_parameters, waypoints, axis=0)

    def generate(
        self,
        obs: dict[str, NDArray[np.floating]],
        include_start: bool = True,
    ) -> tuple[CubicSpline, NDArray[np.floating]]:
        """Génère la trajectoire complète à partir d'une observation.

        Args:
            obs: Dictionnaire d'observation contenant au moins :
                - ``'gates_pos'`` : positions des gates (N, 3)
                - ``'gates_quat'`` : quaternions des gates (N, 4)
                - ``'pos'`` : position courante du drone (utilisée si
                  ``include_start`` est True).
            include_start: Si True, ajoute la position courante du drone comme
                premier waypoint de la trajectoire.

        Returns:
            Tuple ``(spline, waypoints)`` : la cubic spline et les waypoints
            (3 par gate, plus éventuellement le point de départ).
        """
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)

        start_pos = np.asarray(obs["pos"], dtype=float) if include_start else None
        waypoints = self.gate_waypoints(gates_pos, gates_quat, reference_pos=start_pos)

        if include_start:
            waypoints = np.vstack([start_pos, waypoints])

        spline = self.create_spline(waypoints)
        return spline, waypoints

    def sample(
        self,
        spline: CubicSpline,
        num: int = 200,
    ) -> NDArray[np.floating]:
        """Échantillonne la trajectoire pour la visualisation ou le débogage.

        Args:
            spline: Spline générée par :meth:`generate` / :meth:`create_spline`.
            num: Nombre de points d'échantillonnage.

        Returns:
            Points de la trajectoire, shape (num, 3).
        """
        t = np.linspace(0.0, self.duration, num)
        return spline(t)
