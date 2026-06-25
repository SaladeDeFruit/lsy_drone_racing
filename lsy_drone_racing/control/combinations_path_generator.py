"""Generate every candidate path (gate order x gate face) with toppra's spline tool.

4 gates, each with 2 faces (front / back). A combination is an ORDER of the gates (each used
at most once) together with a FACE choice per gate. The number of distinct combinations is:

    N = n_gates! * 2**n_gates      # for 4 gates: 4! * 2**4 = 24 * 16 = 384

For each combination we build the waypoints (start, then per-gate approach/center/exit aligned
on the gate normal), fit a ``toppra.SplineInterpolator`` (the same spline tool used downstream
by TOPP-RA), and collect its coefficients. The spline is purely GEOMETRIC: parameterized by a
normalized arc ``s in [0, 1]``, not time. Time/velocity comes later from TOPP-RA.

The geometry (gates) is the only input: pass gate positions + orientations (from
``obs["gates_pos"]`` / ``obs["gates_quat"]``).
"""

from itertools import permutations, product

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from scipy.spatial.transform import Rotation as R

# Default per-axis kinematic limits for the time-optimal (TOPP-RA) parameterization. Same
# values as the StateController / state_mpc_racing: z is gentler than xy. Format per axis:
# [lower, upper].
DEFAULT_V_MAX_XY = 1.5  # m/s
DEFAULT_V_MAX_Z = 1.0  # m/s
DEFAULT_A_MAX_XY = 3.5  # m/s^2
DEFAULT_A_MAX_Z = 2.0  # m/s^2

GATE_OPENING = 0.4  # m, square gate opening (inner frame), see config/level0.toml


def gate_normals(gates_quat):
    """Gate normals (crossing direction) = gate-frame x-axis expressed in world.

    Args:
        gates_quat: (n_gates, 4) xyzw quaternions, e.g. ``obs["gates_quat"]``.

    Returns:
        (n_gates, 3) UNIT normals (explicitly renormalized so the approach/exit offset is
        exactly ``approach_dist`` meters even if the input quaternions are slightly non-unit).
    """
    normals = np.atleast_2d(R.from_quat(gates_quat).apply([1.0, 0.0, 0.0]))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def gate_face_waypoints(center, normal, face, approach_dist):
    """The 3 waypoints for one gate: approach -> center -> exit, aligned on the gate normal.

    The three points are COLLINEAR along the (unit) gate normal and the middle one is exactly
    the gate center. Forcing three collinear points makes the fitted spline cross the gate
    straight through its center with a tangent aligned on the normal, i.e. a PERPENDICULAR
    crossing (never clipping the frame at an angle).

    Args:
        center: (3,) gate center (``gates_pos[g]``).
        normal: (3,) gate normal (unit; see ``gate_normals``).
        face: 0 -> enter from the front (-normal side), 1 -> enter from the back (+normal side).
        approach_dist: distance of the approach/exit points from the center, in meters.

    Returns:
        list of three (3,) waypoints in traversal order: [approach, center, exit].
    """
    center = np.asarray(center, dtype=float)
    # Defensive: guarantee a unit normal so the offset is exactly approach_dist and the
    # approach/exit points stay perfectly aligned (perpendicular) with the gate center.
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    before = center - approach_dist * normal  # approach point, one side of the gate
    after = center + approach_dist * normal  # exit point, the opposite side
    # Always 3 points, center in the middle; face only flips which side is the approach.
    return [before, center, after] if face == 0 else [after, center, before]


def generate_combinations(n_gates):
    """All (order, faces) combinations: every gate ordering x every per-gate face choice.

    Returns:
        list of (order, faces) tuples. Length = n_gates! * 2**n_gates.
    """
    combos = []
    for order in permutations(range(n_gates)):
        for faces in product((0, 1), repeat=n_gates):
            combos.append((order, faces))
    return combos


def build_waypoints(start_pos, gates_pos, normals, order, faces, approach_dist):
    """Waypoint array for one combination: start + per-gate (approach, center, exit)."""
    waypoints = [np.asarray(start_pos, dtype=float)]
    for g, face in zip(order, faces):
        waypoints.extend(gate_face_waypoints(gates_pos[g], normals[g], face, approach_dist))
    return np.array(waypoints)


def chord_length_param(waypoints):
    """Normalized cumulative chord length in [0, 1] (parameter ~ proportional to distance).

    Use this for the spline knots instead of ``np.linspace`` (uniform). The waypoints are very
    unevenly spaced: the approach/center/exit triplet around each gate is only a few cm apart,
    while consecutive gates are ~1 m apart. With UNIFORM knots every segment gets the same
    parameter step, so the spline must cover a 1 m jump and a 5 cm triplet in equal "time": to
    keep its curvature continuous a CubicSpline then overshoots and curls into a little loop
    right at each gate (the path runs PAST the center and doubles back). Distance-proportional
    knots give each tiny triplet a tiny parameter step, keeping the spline speed consistent and
    removing those loops.
    """
    seglen = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total <= 0:
        return np.linspace(0.0, 1.0, len(waypoints))
    return s / total


def spline_coefficients(path):
    """Extract the cubic-polynomial coefficients from a toppra SplineInterpolator.

    toppra wraps a ``scipy.interpolate.CubicSpline`` in ``path.cspl``; its ``.c`` array has
    shape ``(4, n_segments, 3)`` = (cubic coeffs, segments, xyz axes).
    """
    return np.asarray(path.cspl.c)


def generate_combination_splines(
    start_pos, gates_pos, gates_quat, approach_dist=0.05, bc_type="clamped"
):
    """Build a toppra spline per combination and collect every spline's coefficients.

    Args:
        start_pos: (3,) drone start position (``obs["pos"]``).
        gates_pos: (n_gates, 3) gate centers (``obs["gates_pos"]``).
        gates_quat: (n_gates, 4) xyzw gate orientations (``obs["gates_quat"]``).
        approach_dist: approach/exit offset along the gate normal, in meters.
        bc_type: boundary condition for the spline ("clamped" -> start/end at rest).

    Returns:
        spline_coeffs: list of coefficient arrays, one per combination, each of shape
            ``(4, n_segments, 3)``.
        paths: list of ``toppra.SplineInterpolator`` (ready to feed to TOPP-RA).
        combos: matching list of (order, faces) so each candidate is identifiable.
    """
    gates_pos = np.asarray(gates_pos, dtype=float)
    normals = gate_normals(gates_quat)
    n_gates = len(gates_pos)

    combos = generate_combinations(n_gates)
    spline_coeffs, paths = [], []
    for order, faces in combos:
        waypoints = build_waypoints(start_pos, gates_pos, normals, order, faces, approach_dist)
        ss = chord_length_param(waypoints)  # distance-proportional knots (no gate loops)
        path = ta.SplineInterpolator(ss, waypoints, bc_type=bc_type)
        paths.append(path)
    return paths, combos


def path_duration(path, vbounds, abounds):
    """Time-optimal traversal duration of one geometric path under per-axis kinematic limits.

    Runs TOPP-RA (the same setup as the StateController) on the spline and returns
    ``trajectory.duration``.

    Args:
        path: a ``toppra.SplineInterpolator`` (geometric, arc-parameterized).
        vbounds: (3, 2) per-axis velocity limits [lower, upper].
        abounds: (3, 2) per-axis acceleration limits [lower, upper].

    Returns:
        The traversal time in seconds, or ``None`` if TOPP-RA fails for this path.
    """
    pc_vel = constraint.JointVelocityConstraint(vbounds)
    pc_acc = constraint.JointAccelerationConstraint(abounds)
    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    trajectory = instance.compute_trajectory()
    if trajectory is None:
        return None
    return float(trajectory.duration)


def rank_combinations_by_duration(
    start_pos,
    gates_pos,
    gates_quat,
    approach_dist=0.05,
    bc_type="clamped",
    v_max_xy=DEFAULT_V_MAX_XY,
    v_max_z=DEFAULT_V_MAX_Z,
    a_max_xy=DEFAULT_A_MAX_XY,
    a_max_z=DEFAULT_A_MAX_Z,
):
    """Measure every combination's time-optimal traversal time and rank them (fastest first).

    For each (gate order, per-gate face) combination, builds the geometric spline, runs TOPP-RA
    under the given kinematic limits, and records the resulting duration. Combinations for which
    TOPP-RA fails are skipped.

    Args:
        start_pos: (3,) drone start position (``obs["pos"]``).
        gates_pos: (n_gates, 3) gate centers (``obs["gates_pos"]``).
        gates_quat: (n_gates, 4) xyzw gate orientations (``obs["gates_quat"]``).
        approach_dist: approach/exit offset along the gate normal, in meters.
        bc_type: spline boundary condition ("clamped" -> start/end at rest).
        v_max_xy, v_max_z: velocity limits (m/s) for the xy axes and the z axis.
        a_max_xy, a_max_z: acceleration limits (m/s^2) for the xy axes and the z axis.

    Returns:
        A list of ``(order, faces, duration)`` tuples sorted by ascending ``duration``:
            order: tuple of gate indices in traversal order.
            faces: tuple of per-gate entry faces (0 -> front, 1 -> back).
            duration: traversal time in seconds.
    """
    vbounds = np.array([[-v_max_xy, v_max_xy], [-v_max_xy, v_max_xy], [-v_max_z, v_max_z]])
    abounds = np.array([[-a_max_xy, a_max_xy], [-a_max_xy, a_max_xy], [-a_max_z, a_max_z]])

    paths, combos = generate_combination_splines(
        start_pos, gates_pos, gates_quat, approach_dist, bc_type
    )

    ranked = []
    for (order, faces), path in zip(combos, paths):
        duration = path_duration(path, vbounds, abounds)
        if duration is not None:
            ranked.append((order, faces, duration))

    ranked.sort(key=lambda item: item[2])
    return ranked


def optimal_waypoints(
    start_pos,
    gates_pos,
    gates_quat,
    approach_dist=0.05,
    bc_type="clamped",
    v_max_xy=DEFAULT_V_MAX_XY,
    v_max_z=DEFAULT_V_MAX_Z,
    a_max_xy=DEFAULT_A_MAX_XY,
    a_max_z=DEFAULT_A_MAX_Z,
):
    """Waypoints to follow for the time-optimal path (fastest gate order + faces).

    Ranks every (order, faces) combination by TOPP-RA traversal time and rebuilds the waypoint
    list of the fastest one.

    Args:
        start_pos: (3,) drone start position (``obs["pos"]``).
        gates_pos: (n_gates, 3) gate centers (``obs["gates_pos"]``).
        gates_quat: (n_gates, 4) xyzw gate orientations (``obs["gates_quat"]``).
        approach_dist: approach/exit offset along the gate normal, in meters.
        bc_type: spline boundary condition used when timing the candidates.
        v_max_xy, v_max_z, a_max_xy, a_max_z: per-axis kinematic limits for TOPP-RA.

    Returns:
        (n_waypoints, 3) array of waypoints in traversal order: start, then per-gate
        (approach, center, exit) for the fastest combination.
    """
    ranked = rank_combinations_by_duration(
        start_pos, gates_pos, gates_quat, approach_dist, bc_type,
        v_max_xy, v_max_z, a_max_xy, a_max_z,
    )
    best_order, best_faces, _ = ranked[0]
    normals = gate_normals(gates_quat)
    return build_waypoints(
        start_pos, np.asarray(gates_pos, dtype=float), normals, best_order, best_faces, approach_dist
    )


# =====================================================================================
# Bridge to the TOTG optimizer (TOGTPOptimizer.DroneRacingOptimizer).
#
# combinations_path_generator finds the fastest GATE ORDER; the optimizer then refines the
# exact crossing point inside each gate and the segment times. The optimizer needs the gates
# as POLYGONS (4 vertices) and a properly configured drone model.
# =====================================================================================


def gate_polygon(center, quat, opening=GATE_OPENING):
    """The 4 corner vertices of a gate opening (square ``opening`` x ``opening``).

    The square lies in the plane perpendicular to the gate normal (the gate-frame y/z axes),
    centered on the gate center. This is what ``TOGTPOptimizer.PolygonGate`` expects.

    Args:
        center: (3,) gate center (``gates_pos[g]``).
        quat: (4,) xyzw gate orientation (``gates_quat[g]``).
        opening: side length of the square opening, in meters.

    Returns:
        list of four (3,) corner positions.
    """
    rot = R.from_quat(quat)
    y = rot.apply([0.0, 1.0, 0.0])  # gate width axis (world)
    z = rot.apply([0.0, 0.0, 1.0])  # gate height axis (world)
    h = 0.5 * opening
    c = np.asarray(center, dtype=float)
    return [c + h * y + h * z, c + h * y - h * z, c - h * y - h * z, c - h * y + h * z]


def cf2x_p250_model():
    """A ``QuadrotorModel`` configured for the cf2x_P250.

    The optimizer's default ``QuadrotorModel`` is a large generic quad (mass 0.85 kg,
    f_max 6.88 N) — wrong for our drone, which would make the thrust constraint meaningless.
    Only mass / gravity / f_max enter the current cost; the rest is set for completeness.
    """
    from lsy_drone_racing.control.TOGTPOptimizer import QuadrotorModel

    model = QuadrotorModel()
    model.mass = 0.0318  # kg
    model.f_max = 0.12  # N per motor (thrust_max)
    model.gravity = np.array([0.0, 0.0, -9.81])
    model.arm_length = 0.046  # m (cf2x approx)
    model.inertia = np.array([16.8e-6, 16.8e-6, 29.8e-6])  # kg m^2
    return model


def build_optimizer(
    start_pos,
    gates_pos,
    gates_quat,
    drone_model=None,
    end_pos=None,
    opening=GATE_OPENING,
    exit_dist=0.5,
    approach_dist=0.05,
    v_max_xy=DEFAULT_V_MAX_XY,
    v_max_z=DEFAULT_V_MAX_Z,
    a_max_xy=DEFAULT_A_MAX_XY,
    a_max_z=DEFAULT_A_MAX_Z,
):
    """Rank the combinations, then build a ready-to-solve ``DroneRacingOptimizer``.

    Sends the optimizer everything it needs: the drone model (cf2x_P250 by default) and the
    gates as ``PolygonGate`` in the time-optimal order, plus the start/end positions.

    Args:
        start_pos: (3,) drone start (``obs["pos"]``).
        gates_pos: (n_gates, 3) gate centers (``obs["gates_pos"]``).
        gates_quat: (n_gates, 4) xyzw gate orientations (``obs["gates_quat"]``).
        drone_model: a ``QuadrotorModel``; defaults to :func:`cf2x_p250_model`.
        end_pos: (3,) final position; defaults to a point ``exit_dist`` past the last gate,
            along its exit direction (from the chosen face).
        opening: gate opening side length (m), for the polygon vertices.
        exit_dist: how far past the last gate the default end point sits (m).
        approach_dist, v_max_*, a_max_*: passed to the combination ranking (TOPP-RA timing).

    Returns:
        (optimizer, best_order, best_duration):
            optimizer: a ``DroneRacingOptimizer`` ready for ``.solve()``.
            best_order: tuple of gate indices in the chosen traversal order.
            best_duration: TOPP-RA traversal time (s) of that order (the ranking estimate).
    """
    from lsy_drone_racing.control.TOGTPOptimizer import DroneRacingOptimizer, PolygonGate

    gates_pos = np.asarray(gates_pos, dtype=float)

    # 1. Fastest gate order (and faces) from the TOPP-RA combination ranking.
    ranked = rank_combinations_by_duration(
        start_pos, gates_pos, gates_quat, approach_dist, "clamped",
        v_max_xy, v_max_z, a_max_xy, a_max_z,
    )
    best_order, best_faces, best_duration = ranked[0]

    # 2. The needed gates, as PolygonGate, IN THE OPTIMAL ORDER.
    gates = [
        PolygonGate(seq, gate_polygon(gates_pos[g], gates_quat[g], opening))
        for seq, g in enumerate(best_order)
    ]

    # 3. The drone model (cf2x_P250 by default).
    model = drone_model if drone_model is not None else cf2x_p250_model()

    # 4. Start / end positions. Default end = past the last gate, in its exit direction.
    if end_pos is None:
        normals = gate_normals(gates_quat)
        last, last_face = best_order[-1], best_faces[-1]
        exit_dir = normals[last] if last_face == 0 else -normals[last]
        end_pos = gates_pos[last] + exit_dist * exit_dir

    optimizer = DroneRacingOptimizer(
        model, gates, np.asarray(start_pos, dtype=float), np.asarray(end_pos, dtype=float)
    )
    return optimizer, best_order, best_duration


def solve_optimal(start_pos, gates_pos, gates_quat, **kwargs):
    """Build the optimizer for the fastest gate order and run ``.solve()``.

    Returns:
        (result, optimizer, best_order): ``result`` is the scipy ``OptimizeResult``.
    """
    optimizer, best_order, _ = build_optimizer(start_pos, gates_pos, gates_quat, **kwargs)
    return optimizer.solve(), optimizer, best_order


if __name__ == "__main__":
    # Demo on the level0 nominal gates (replace with obs["gates_pos"]/["gates_quat"] at runtime).
    start = np.array([-1.5, 0.75, 0.01])
    gpos = np.array([[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]])
    grpy = np.array([[0, 0, -0.78], [0, 0, 2.35], [0, 0, 3.14], [0, 0, 0.0]])
    gquat = R.from_euler("xyz", grpy).as_quat()

    # 1. Fastest gate order, then hand everything to the TOTG optimizer and solve.
    result, optimizer, best_order = solve_optimal(start, gpos, gquat)
    print(f"Best gate order: {best_order}")
    print(f"Optimizer success: {result.success} | final cost (total time + penalty): {result.fun:.3f}")