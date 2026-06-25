""" 
Path Planning Module for Drone Racing 

Spline-based planner : 
- Trajectoire = spline d'Hermite QUINTIQUE (degré 5) passant par les centres de portes, 
  avec tangente = normale de la porte et courbure = 0 au croisement (passage centré et 
  perpendiculaire, un seul nœud par porte). 
- Évitement d'obstacles clairsemé (détection 3D, push radial), sans densification. 
- Sécurité cadre : bande radiale dans le plan de la porte (push d'un nœud vers le centre). 
""" 

from __future__ import annotations 

from typing import TYPE_CHECKING, List, Tuple, Optional 
from dataclasses import dataclass, field 
from pathlib import Path 

import numpy as np 
from scipy.interpolate import CubicSpline, BPoly 
from scipy.spatial.transform import Rotation 

if TYPE_CHECKING: 
    from numpy.typing import NDArray 


# ============================================================================= 
# Data Classes for Configuration 
# ============================================================================= 

@dataclass 
class PathConfig: 
    """Configuration for path planning."""
    # Hermite spline
    tension: float = 1.0                     # Échelle des tangentes imposées (portes/start/exit)
    tangent_chord_scale: bool = False        # Tangentes imposées ∝ corde locale (sinon unitaires)
    clamp_curvature: bool = True             # True=quintique (courbure=0 aux nœuds), False=cubique C1
    gate_tangent_flow_blend: float = 0.6     # 0=normale pure ; >0 incline la tangente vers le flux (lisse les angles)
    max_gate_obliquity: float = 0.7          # Angle max (rad) entre tangente porte et NORMALE (~40°) :
                                             # garde le croisement assez perpendiculaire pour dégager le cadre
    exit_distance: float = 0.3               # Distance du nœud de sortie après la dernière porte

    # ---- Évitement du cadre des portes (anneau de 16 cm autour de l'ouverture de 40 cm) ----
    # Géométrie réelle (gate.xml, repère porte : x=normale, y=largeur, z=hauteur) :
    #   ouverture = ±0.20 m ; anneau plein = 0.20 -> 0.36 m ; épaisseur ±0.01 m le long de x.
    # Deux cas distincts :
    #   - porte à TRAVERSER : on ramène le croisement DANS la fenêtre d'ouverture (intérieur).
    #   - porte NON visitée à cet instant : on contourne l'anneau par l'EXTÉRIEUR.
    avoid_gate_frames: bool = True           # RÉACTIVÉ: Le retour en arc large du hairpin protège le tracé.
    gate_half_opening: float = 0.2           # Demi-ouverture (0.40 m)
    frame_band_outer: float = 0.36           # Demi-extension extérieure de l'anneau
    frame_plane_halfthick: float = 0.01      # Demi-épaisseur de l'anneau le long de la normale
    frame_drone_radius: float = 0.08         # Demi-largeur effective du drone (collision cadre)
    frame_inner_margin: float = 0.04         # Marge intérieure (passage centré)
    frame_outer_margin: float = 0.05         # Marge extérieure (contournement)
    frame_intended_window: float = 0.35      # Demi-fenêtre d'arc autour du centre = croisement « voulu »
    drone_radius: float = 0.12               # (conservé pour compat ; non utilisé par l'évitement cadre)
    frame_margin: float = 0.1                # (idem)

    # Obstacle avoidance 
    safety_distance: float = 0.6             # Minimum distance from obstacles (meters) 

    # Arc-length reparameterization 
    arc_step: float = 0.05                   # Arc length sampling step 
    arc_epsilon: float = 1e-5                # Convergence threshold 

    # Trajectory extension 
    extend_length: float = 0.1               # Extension length at trajectory end 


@dataclass 
class TrajectoryResult: 
    """Result of trajectory planning.""" 
    spline: object                           # Main trajectory spline (BPoly quintique) 
    arc_spline: Optional[CubicSpline]        # Arc-length parameterized spline 
    waypoints: NDArray[np.floating]          # Original waypoints (nœuds) 
    total_length: float                      # Total arc length 
    gate_thetas: Optional[NDArray]           # Arc-length parameters at gates 
    gate_positions: Optional[NDArray] = None # Gate positions 
    gate_normals: Optional[NDArray] = None   # Gate normals 
    obstacle_positions: Optional[NDArray] = None # Obstacle positions 
    trajectory_duration: float = 30.0        # Trajectory duration 


# ============================================================================= 
# Composite Spline for Multi-Stage Trajectories 
# ============================================================================= 

class CompositeSpline: 
    """Composite spline combining two trajectory segments.""" 

    def __init__(self, first: CubicSpline, second: CubicSpline, offset: float): 
        self.trajectory_1 = first 
        self.trajectory_2 = second 
        self.offset = offset 
        self.x = np.concatenate([first.x, second.x + offset]) 

    def __call__(self, t): 
        if np.isscalar(t): 
            return self.trajectory_1(t) if t < self.offset else self.trajectory_2(t - self.offset) 
        return np.array([self(t_i) for t_i in t]) 

    def derivative(self, order: int = 1): 
        return CompositeSpline( 
            self.trajectory_1.derivative(order), 
            self.trajectory_2.derivative(order), 
            self.offset, 
        ) 


# ============================================================================= 
# Gate Frame Utilities 
# ============================================================================= 

class GateFrameExtractor: 
    """Utility class for extracting gate coordinate frames from quaternions.""" 

    @staticmethod 
    def extract_frames( 
        gates_quaternions: NDArray[np.floating] 
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: 
        """ 
        Extract complete local coordinate frames for each gate. 

        Returns (normals, y_axes, z_axes): 
        - normals: gate normal (x-axis, penetration direction) 
        - y_axes: gate width direction 
        - z_axes: gate height direction 
        """ 
        rotations = Rotation.from_quat(gates_quaternions) 
        rotation_matrices = rotations.as_matrix() 

        normals = rotation_matrices[:, :, 0]   # First column: normal (x-axis) 
        y_axes = rotation_matrices[:, :, 1]    # Second column: width (y-axis) 
        z_axes = rotation_matrices[:, :, 2]    # Third column: height (z-axis) 

        return normals, y_axes, z_axes 

    @staticmethod 
    def extract_normals(gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]: 
        """Extract only gate normal vectors.""" 
        rotations = Rotation.from_quat(gates_quaternions) 
        rotation_matrices = rotations.as_matrix() 
        return rotation_matrices[:, :, 0] 


# ============================================================================= 
# Path Planning Core 
# ============================================================================= 

def _unit(v: np.ndarray) -> np.ndarray: 
    """Return the unit vector of v (safe for ~zero vectors).""" 
    n = np.linalg.norm(v) 
    return v / n if n > 1e-9 else np.zeros_like(v) 


class PathPlanner: 
    """ 
    Spline-based path planner (Hermite quintique + évitement obstacles/cadre). 
    """ 

    def __init__(self, config: Optional[PathConfig] = None): 
        self.config = config or PathConfig() 
        self._debug_info = {} 

    # ========================================================================= 
    # Représentation des nœuds + construction de la spline quintique 
    # ========================================================================= 
    # 
    # Un nœud est un dict : 
    #   {'pos': (3,), 'dir': (3,) unit ou None, 'is_gate': bool, 'normal': (3,) ou None} 
    # - 'dir' imposé (porte/start/exit) -> tangente = dir * tension ; sinon Catmull-Rom. 
    # - courbure (2e dérivée) = 0 à TOUS les nœuds -> spline quintique C2, localement droite 
    #   aux portes (donc passage centré/perpendiculaire dégageant le cadre). 

    @staticmethod 
    def _node(pos, dir=None, is_gate=False, normal=None) -> dict: 
        return { 
            'pos': np.asarray(pos, dtype=float), 
            'dir': None if dir is None else _unit(np.asarray(dir, dtype=float)), 
            'is_gate': bool(is_gate), 
            'normal': None if normal is None else np.asarray(normal, dtype=float), 
        } 

    def _build_from_nodes(self, nodes: List[dict]) -> BPoly: 
        """Construit la spline d'Hermite quintique à partir de la liste de nœuds.""" 
        pos = np.array([n['pos'] for n in nodes], dtype=float)        # (N, 3) 

        # Paramètre = longueur d'arc cumulée (corde). Filtre les doublons consécutifs. 
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) 
        keep = np.concatenate([[True], seg > 1e-6]) 
        if not keep.all(): 
            nodes = [n for n, k in zip(nodes, keep) if k] 
            pos = pos[keep] 
            seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) 
        x = np.concatenate([[0.0], np.cumsum(seg)]) 

        N = len(nodes)
        tang = np.zeros((N, 3))
        tension = self.config.tension
        # Échelle locale de la tangente : corde locale (moyenne des segments adjacents).
        # Une tangente ∝ corde répartit la courbure sur tout le segment (lissage) au lieu
        # de la concentrer en un pic au nœud (kink).
        if self.config.tangent_chord_scale:
            chord = np.empty(N)
            for i in range(N):
                left = x[i] - x[i - 1] if i > 0 else None
                right = x[i + 1] - x[i] if i < N - 1 else None
                vals = [v for v in (left, right) if v is not None]
                chord[i] = float(np.mean(vals)) if vals else 1.0
        else:
            chord = np.ones(N)
        for i, n in enumerate(nodes):
            if n['dir'] is not None:                         # tangente imposée
                tang[i] = n['dir'] * tension * chord[i]
            elif 0 < i < N - 1:                              # Catmull-Rom (nœud libre)
                tang[i] = (pos[i + 1] - pos[i - 1]) / (x[i + 1] - x[i - 1] + 1e-9) 
            elif i == 0: 
                tang[i] = (pos[1] - pos[0]) / (x[1] - x[0] + 1e-9) 
            else: 
                tang[i] = (pos[-1] - pos[-2]) / (x[-1] - x[-2] + 1e-9) 

        if self.config.clamp_curvature:
            curv = np.zeros((N, 3))                          # courbure nulle -> quintique C2
            yi = np.stack([pos, tang, curv], axis=1)         # (N, pos/tang/curv, 3 dims) -> degré 5
        else:
            yi = np.stack([pos, tang], axis=1)               # (N, pos/tang, 3 dims) -> cubique C1
        return BPoly.from_derivatives(x, yi)

    def _insert_free_node(self, nodes: List[dict], theta: float, new_pos: np.ndarray) -> List[dict]: 
        """Insère un nœud libre à la position d'arc ``theta`` (entre les 2 nœuds encadrants).""" 
        pos = np.array([n['pos'] for n in nodes], dtype=float) 
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) 
        x = np.concatenate([[0.0], np.cumsum(seg)]) 
        i = int(np.clip(np.searchsorted(x, theta) - 1, 0, len(nodes) - 2)) 
        nodes.insert(i + 1, self._node(new_pos)) 
        return nodes 

    # ========================================================================= 
    # Évitement d'obstacles clairsemé (détection 3D, push radial) 
    # ========================================================================= 

    def _avoid_obstacles_sparse( 
        self, 
        spline: BPoly, 
        nodes: List[dict], 
        obstacles: NDArray[np.floating], 
    ) -> Tuple[BPoly, List[dict]]: 
        """Insère ≤1 nœud d'évitement par collision réelle (détection 3D), sans densifier.""" 
        if obstacles is None or len(obstacles) == 0: 
            return spline, nodes 

        safe = self.config.safety_distance 
        obstacles = np.asarray(obstacles, dtype=float) 

        for _pass in range(2): 
            total = float(spline.x[-1]) 
            ts = np.arange(0.0, total, self.config.arc_step) 
            pts = np.asarray(spline(ts)) 
            inserted = False 

            for obs in obstacles: 
                d = np.linalg.norm(pts - obs[:3], axis=1)           # distance 3D 
                inside = d < safe 
                if not inside.any(): 
                    continue 

                # Groupes d'indices contigus (runs de collision) 
                idx = np.where(inside)[0] 
                splits = np.where(np.diff(idx) > 1)[0] + 1 
                for run in np.split(idx, splits): 
                    entry, exit_ = pts[run[0]], pts[run[-1]] 
                    direction = (entry - obs[:3]) + (exit_ - obs[:3]) 
                    direction = _unit(direction) 
                    if np.linalg.norm(direction) < 1e-9: 
                        direction = np.array([0.0, 0.0, 1.0])       # poussée verticale par défaut 
                    new_pos = obs[:3] + direction
                    theta_mid = ts[run[len(run) // 2]] 
                    nodes = self._insert_free_node(nodes, theta_mid, new_pos) 
                    inserted = True 

            if not inserted: 
                break 
            spline = self._build_from_nodes(nodes) 

        return spline, nodes 

    # =========================================================================
    # Évitement du cadre des portes (anneau de 16 cm) — deux cas distincts
    # =========================================================================

    def _avoid_gate_frames(
        self,
        spline: BPoly,
        nodes: List[dict],
        gate_centers: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
    ) -> Tuple[BPoly, List[dict]]:
        """Empêche le tracé de heurter l'anneau plein autour de l'ouverture (0.20 -> 0.36 m).

        Pour chaque porte, on échantillonne le tracé et on ne regarde que les points proches
        du PLAN de la porte (|x_local| < épaisseur + rayon drone). En coordonnées dans le plan
        (axes largeur/hauteur), ``lat = max(|y|, |z|)`` (anneau carré). Deux cas :

        - **Porte traversée** (le tracé passe par son ouverture, ``min lat`` proche-plan < bord
          d'ouverture) : toute intrusion dans le cadre est RAMENÉE vers l'INTÉRIEUR (y compris le
          demi-tour d'un hairpin qui repasse près du plan de sa propre porte).
        - **Porte non visitée** (le tracé ne passe pas par son ouverture) : on contourne par
          l'EXTÉRIEUR (au-delà de ``frame_band_outer``).
        """
        cfg = self.config
        r = cfg.frame_drone_radius
        half = cfg.gate_half_opening
        safe_inner = max(0.02, half - cfg.frame_inner_margin - r)
        band_lo = half - r                                  # lat où le drone touche l'anneau
        band_hi = cfg.frame_band_outer + r
        exterior = cfg.frame_band_outer + cfg.frame_outer_margin + r
        plane_clear = cfg.frame_plane_halfthick + r         # distance normale pour dégager la bande

        for _pass in range(5):
            total = float(spline.x[-1])
            ts = np.arange(0.0, total, self.config.arc_step)
            pts = np.asarray(spline(ts))
            node_x = np.array([0.0] + list(np.cumsum(
                np.linalg.norm(np.diff([n['pos'] for n in nodes], axis=0), axis=1))))
            corrected = False

            for g in range(len(gate_centers)):
                c = np.asarray(gate_centers[g], dtype=float)
                n = _unit(np.asarray(gate_normals[g], dtype=float))
                u = _unit(np.asarray(gate_y_axes[g], dtype=float))
                v = _unit(np.asarray(gate_z_axes[g], dtype=float))

                e = pts - c
                d_normal = e @ n
                a = e @ u
                b = e @ v
                lat = np.maximum(np.abs(a), np.abs(b))        # anneau carré -> norme L-inf
                dist3 = np.linalg.norm(e, axis=1)
                near = np.abs(d_normal) < plane_clear
                if not near.any():
                    continue

                idx_center = int(np.argmin(dist3))            # point de passage le plus proche du centre

                idx, push_to = None, None
                if near[idx_center] and lat[idx_center] > safe_inner:
                    # cas 1 : le croisement lui-même est excentré -> ramener vers l'INTÉRIEUR
                    # (passage centré dans l'ouverture, p.ex. porte randomisée décalée).
                    idx, push_to = idx_center, safe_inner
                else:
                    # cas 2 : tout point proche-plan QUI RASE l'anneau (croisement déjà centré, ou
                    # demi-tour de hairpin qui repasse près du plan, ou porte non visitée) ->
                    # contourner par l'EXTÉRIEUR. La re-traversée du plan se fait alors au-delà du
                    # cadre (lat > 0.36) au lieu de DANS l'anneau (cas G2 : re-traversée à lat≈0.29).
                    in_band = np.where(near & (lat > band_lo) & (lat < band_hi))[0]
                    if in_band.size:
                        idx, push_to = int(in_band[np.argmin(np.abs(d_normal[in_band]))]), exterior
                if idx is None:
                    continue

                # n'insère pas un nœud quasi-confondu avec un nœud existant (évite les cusps)
                if np.min(np.abs(node_x - ts[idx])) < 0.06:
                    continue

                e_plane = e[idx] - d_normal[idx] * n
                m = max(abs(a[idx]), abs(b[idx]))
                if m < 1e-6:                                  # sur l'axe central : direction arbitraire
                    e_plane = u * 1e-3
                    m = 1e-3
                new_pos = c + d_normal[idx] * n + e_plane * (push_to / m)
                nodes = self._insert_free_node(nodes, ts[idx], new_pos)
                corrected = True

            if not corrected:
                break
            spline = self._build_from_nodes(nodes)

        return spline, nodes

    # ========================================================================= 
    # Spline utilities 
    # ========================================================================= 

    def reparametrize_by_arclength( 
        self, 
        trajectory, 
        arc_step: Optional[float] = None, 
        epsilon: Optional[float] = None 
    ) -> CubicSpline: 
        """Reparametrize trajectory by arc length for uniform speed.""" 
        step = arc_step or self.config.arc_step 
        eps = epsilon or self.config.arc_epsilon 

        total_param_range = trajectory.x[-1] - trajectory.x[0] 

        for _ in range(99): 
            n_segments = max(2, int(total_param_range / step)) 
            t_samples = np.linspace(0.0, total_param_range, n_segments) 
            pts = np.asarray(trajectory(t_samples)) 
            deltas = np.diff(pts, axis=0) 
            seg_lengths = np.linalg.norm(deltas, axis=1) 
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)]) 
            total_param_range = float(cum_arc[-1]) 
            trajectory = CubicSpline(cum_arc, pts) 

            if np.std(seg_lengths) <= eps: 
                return trajectory 

        return trajectory 

    def extend_spline( 
        self, 
        trajectory: CubicSpline, 
        extend_length: Optional[float] = None 
    ) -> CubicSpline: 
        """Extend trajectory along its terminal tangent direction.""" 
        ext_len = extend_length or self.config.extend_length 

        base_knots = trajectory.x 
        base_dt = min(base_knots[1] - base_knots[0], 0.2) 
        p_end = trajectory(base_knots[-1]) 
        v_end = trajectory.derivative(1)(base_knots[-1]) 
        v_dir = v_end / (np.linalg.norm(v_end) + 1e-6) 

        extra_knots = np.arange( 
            base_knots[-1] + base_dt, 
            base_knots[-1] + ext_len, 
            base_dt, 
        ) 
        p_extend = np.array([p_end + v_dir * (s - base_knots[-1]) for s in extra_knots]) 

        theta_new = np.concatenate([base_knots, extra_knots]) 
        p_new = np.vstack([trajectory(base_knots), p_extend]) 

        return CubicSpline(theta_new, p_new, axis=0) 

    def compute_curvature( 
        self, 
        spline: CubicSpline, 
        t_vals: NDArray[np.floating], 
        eps: float = 1e-8 
    ) -> NDArray[np.floating]: 
        """Compute curvature along the spline.""" 
        v = spline(t_vals, 1) 
        a = spline(t_vals, 2) 
        cross_term = np.cross(v, a) 
        num = np.linalg.norm(cross_term, axis=1) 
        den = np.linalg.norm(v, axis=1) ** 3 + eps 
        return num / den 

    def find_closest_point( 
        self, 
        trajectory, 
        position: NDArray[np.floating], 
        sample_interval: float = 0.05 
    ) -> Tuple[float, NDArray[np.floating]]: 
        """Find the closest point on trajectory to a given position.""" 
        total_length = float(trajectory.x[-1]) 
        t_samples = np.arange(0.0, total_length, sample_interval) 
        if t_samples.size == 0: 
            return 0.0, np.asarray(trajectory(0.0)) 

        points = np.asarray(trajectory(t_samples)) 
        dists = np.linalg.norm(points - position, axis=1) 
        idx_min = int(np.argmin(dists)) 

        return idx_min * sample_interval, points[idx_min] 

    def get_gate_parameters( 
        self, 
        trajectory, 
        gate_positions: NDArray[np.floating], 
        sample_interval: float = 0.05 
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]: 
        """Get arc-length parameters corresponding to gate positions.""" 
        theta_list = [] 
        pos_list = [] 

        for gate_center in gate_positions: 
            theta, pos = self.find_closest_point(trajectory, gate_center, sample_interval) 
            theta_list.append(theta) 
            pos_list.append(pos) 

        return np.array(theta_list), np.array(pos_list) 

    # ========================================================================= 
    # Complete Path Planning Pipeline 
    # ========================================================================= 

    def _gate_passage_tangent(
        self,
        n_signed: np.ndarray,
        prev: np.ndarray,
        center: np.ndarray,
        nxt: np.ndarray,
    ) -> np.ndarray:
        """Tangente de passage à une porte : normale signée inclinée vers le flux local.

        Le flux est la bissectrice des cordes entrante (prev->center) et sortante
        (center->nxt). On mélange ``normale`` et ``flux`` (``gate_tangent_flow_blend``) pour
        lisser, MAIS on borne l'**obliquité** : l'angle entre la tangente et la NORMALE reste
        ≤ ``max_gate_obliquity``. Un croisement trop oblique ferait glisser le tracé le long du
        plan de la porte et raserait l'anneau du cadre (cas observé à G2, où le flux est presque
        parallèle au plan de la porte). En gardant un minimum de composante normale, le tracé
        s'écarte du plan (donc du cadre mince) avant de virer latéralement.
        """
        blend = float(self.config.gate_tangent_flow_blend)
        if blend <= 0.0:
            return n_signed

        flow = _unit(_unit(center - prev) + _unit(nxt - center))
        if np.linalg.norm(flow) < 1e-9:
            return n_signed
        # garde le flux dans le demi-espace de passage (même sens que la normale)
        if np.dot(flow, n_signed) < 0.0:
            flow = -flow

        tangent = _unit((1.0 - blend) * n_signed + blend * flow)
        # borne l'obliquité : tangente à au plus ``max_gate_obliquity`` de la normale
        obl = float(self.config.max_gate_obliquity)
        if np.dot(tangent, n_signed) < np.cos(obl):
            perp = _unit(tangent - np.dot(tangent, n_signed) * n_signed)
            tangent = _unit(np.cos(obl) * n_signed + np.sin(obl) * perp)
        return tangent

    def _build_gate_nodes(
        self,
        initial_pos: np.ndarray,
        gate_centers: np.ndarray,
        gate_normals: np.ndarray,
    ) -> Tuple[List[dict], np.ndarray]:
        """Construit la liste de nœuds [start, centres…, exit] avec tangentes imposées 
        (normale orientée dans le sens de passage). Retourne aussi les normales signées.""" 
        n_gates = len(gate_centers) 
        nodes: List[dict] = [] 
        signed_normals = np.zeros((n_gates, 3)) 

        # start : tangente vers la première porte 
        nodes.append(self._node(initial_pos, dir=(gate_centers[0] - initial_pos))) 

        prev = np.asarray(initial_pos, dtype=float)
        for k in range(n_gates):
            c = np.asarray(gate_centers[k], dtype=float)
            nk = _unit(np.asarray(gate_normals[k], dtype=float))
            nxt = np.asarray(gate_centers[k + 1] if k + 1 < n_gates else (c + nk), dtype=float)
            
            # 1. Détection du hairpin (via les vraies positions des portes voisines)
            true_prev = np.asarray(gate_centers[k - 1] if k > 0 else initial_pos, dtype=float)
            s_prev = np.sign(np.dot(true_prev - c, nk))
            s_next = np.sign(np.dot(nxt - c, nk))
            is_hairpin = (s_prev == s_next) and (s_prev != 0)

            # 2. Sens de passage (n_signed)
            if is_hairpin:
                s = -s_prev  # Traversée forcée face à l'entrée
            else:
                travel = nxt - prev
                s = 1.0 if np.dot(nk, travel) >= 0.0 else -1.0
                
            n_signed = s * nk
            signed_normals[k] = n_signed
            
            # Tangente de passage calculée et bornée
            tangent_dir = self._gate_passage_tangent(n_signed, prev, c, nxt)
            nodes.append(self._node(c, dir=tangent_dir, is_gate=True, normal=n_signed))
            
            # 3. Insertion du waypoint de retour (si hairpin)
            if is_hairpin:
                d_n = 0.25       # Éloignement normal pour dégager le cadre mince
                lat_apex = 0.50  # Déplacement latéral au-delà de l'anneau (0.36m)
                side = -s_next   # Côté opposé aux voisines (ressortie)
                
                # Projection de nxt dans le plan de la porte pour définir la direction latérale
                to_nxt = nxt - c
                nxt_plane = to_nxt - np.dot(to_nxt, nk) * nk
                if np.linalg.norm(nxt_plane) > 1e-6:
                    u_lat = _unit(nxt_plane)
                else:
                    u_lat = _unit(np.cross(nk, [0, 0, 1]))
                    
                apex_pos = c + (side * d_n) * nk + lat_apex * u_lat
                
                # Tangente dirigée (bissectrice porte->apex et apex->nxt)
                v1 = _unit(apex_pos - c)
                v2 = _unit(nxt - apex_pos)
                apex_dir = _unit(v1 + v2)
                
                nodes.append(self._node(apex_pos, dir=apex_dir, is_gate=False))
                
                # L'apex devient le point précédent pour la géométrie de la porte k+1
                prev = apex_pos
            else:
                prev = c

        # exit : un peu après la dernière porte, le long de sa normale signée 
        exit_pos = gate_centers[-1] + self.config.exit_distance * signed_normals[-1] 
        nodes.append(self._node(exit_pos, dir=signed_normals[-1])) 

        return nodes, signed_normals 

    def plan_trajectory( 
        self, 
        obs: dict[str, NDArray[np.floating]], 
        trajectory_duration: float = 30.0, 
        sampling_freq: float = 100.0, 
        for_mpcc: bool = True, 
        mpcc_extension_length: float = 12.0 
    ) -> TrajectoryResult: 
        """Pipeline complet : Hermite quintique -> évitement obstacles -> sécurité cadre -> 
        reparamétrage par longueur d'arc.""" 
        initial_pos = np.asarray(obs['pos'], dtype=float) 
        gate_centers = np.asarray(obs['gates_pos'], dtype=float) 
        gate_quats = np.asarray(obs['gates_quat'], dtype=float) 
        obstacle_positions = np.asarray(obs.get('obstacles_pos', []), dtype=float) 

        gate_normals, gate_y_axes, gate_z_axes = GateFrameExtractor.extract_frames(gate_quats)

        # 1. Spline d'Hermite quintique par [start, centres, exit]
        nodes, signed_normals = self._build_gate_nodes(initial_pos, gate_centers, gate_normals)
        spline = self._build_from_nodes(nodes)

        # 2. Évitement d'obstacles (push away, détection 3D)
        spline, nodes = self._avoid_obstacles_sparse(spline, nodes, obstacle_positions)

        # 3. Évitement du cadre des portes : intérieur si on traverse, extérieur sinon
        if self.config.avoid_gate_frames:
            spline, nodes = self._avoid_gate_frames(
                spline, nodes, gate_centers, gate_normals, gate_y_axes, gate_z_axes)

        node_positions = np.array([n['pos'] for n in nodes]) 

        # 4. Reparamétrage par longueur d'arc (pour le MPCC : theta en mètres) 
        arc_spline = None 
        gate_thetas = None 
        total_length = float(spline.x[-1]) 

        if for_mpcc: 
            arc_spline = self.reparametrize_by_arclength(spline) 
            total_length = float(arc_spline.x[-1]) 
            gate_thetas, _ = self.get_gate_parameters(arc_spline, gate_centers) 

        return TrajectoryResult( 
            spline=spline, 
            arc_spline=arc_spline, 
            waypoints=node_positions, 
            total_length=total_length, 
            gate_thetas=gate_thetas, 
            gate_positions=gate_centers, 
            gate_normals=gate_normals, 
            obstacle_positions=obstacle_positions, 
            trajectory_duration=trajectory_duration 
        ) 

    def replan_trajectory( 
        self, 
        obs: dict[str, NDArray[np.floating]], 
        current_position: NDArray[np.floating], 
        **kwargs 
    ) -> TrajectoryResult: 
        """Replan trajectory from current position.""" 
        obs = obs.copy() 
        obs['pos'] = current_position 
        return self.plan_trajectory(obs, **kwargs)