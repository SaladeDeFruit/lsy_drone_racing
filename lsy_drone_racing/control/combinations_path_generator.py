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

import math
from itertools import permutations, product

import numpy as np
import toppra as ta
from scipy.spatial.transform import Rotation as R


def gate_normals(gates_quat):
    """Gate normals (crossing direction) = gate-frame x-axis expressed in world.

    Args:
        gates_quat: (n_gates, 4) xyzw quaternions, e.g. ``obs["gates_quat"]``.

    Returns:
        (n_gates, 3) unit normals.
    """
    return R.from_quat(gates_quat).apply([1.0, 0.0, 0.0])


def gate_face_waypoints(center, normal, face, approach_dist):
    """Approach / center / exit waypoints for one gate, given the entry face.

    Args:
        center: (3,) gate center.
        normal: (3,) gate normal.
        face: 0 -> enter from the front (-normal side), 1 -> enter from the back (+normal side).
        approach_dist: distance of the approach/exit points from the center, in meters.

    Returns:
        list of three (3,) waypoints in traversal order.
    """
    before = center - approach_dist * normal
    after = center + approach_dist * normal
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


def spline_coefficients(path):
    """Extract the cubic-polynomial coefficients from a toppra SplineInterpolator.

    toppra wraps a ``scipy.interpolate.CubicSpline`` in ``path.cspl``; its ``.c`` array has
    shape ``(4, n_segments, 3)`` = (cubic coeffs, segments, xyz axes).
    """
    return np.asarray(path.cspl.c)


def generate_combination_splines(
    start_pos, gates_pos, gates_quat, approach_dist=0.3, bc_type="clamped"
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
        ss = np.linspace(0.0, 1.0, len(waypoints))  # normalized arc parameter
        path = ta.SplineInterpolator(ss, waypoints, bc_type=bc_type)
        paths.append(path)
        spline_coeffs.append(spline_coefficients(path))
    return spline_coeffs, paths, combos


if __name__ == "__main__":
    # Demo on the level0 nominal gates (replace with obs["gates_pos"]/["gates_quat"] at runtime).
    start = np.array([-1.5, 0.75, 0.01])
    gpos = np.array([[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]])
    grpy = np.array([[0, 0, -0.78], [0, 0, 2.35], [0, 0, 3.14], [0, 0, 0.0]])
    gquat = R.from_euler("xyz", grpy).as_quat()

    coeffs, paths, combos = generate_combination_splines(start, gpos, gquat)
    n = len(gpos)
    print(f"n_gates = {n}")
    print(f"combinations = n! * 2^n = {math.factorial(n)} * {2 ** n} = {len(combos)}")
    print(f"toppra splines built = {len(paths)} | coeff sets = {len(coeffs)}")
    print(f"one coeff array shape = {coeffs[0].shape}  (4 cubic coeffs, n_segments, 3 axes)")
