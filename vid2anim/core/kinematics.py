"""Advanced Kinematics and SO(3) Lie Algebra for Skeletal Retargeting.

Ported and generalized from reference production pipelines (Motion2MixamoRig / GVHMR).
Provides:
- Shortest-path Rodrigues swing rotation aligning
- Orthonormal basis frame construction
- Continuous quaternion tracking (no antipodal flips / sign pops)
- Anti-twist 180-degree geodesic resolution
- Bidirectional SO(3) matrix <-> glTF quaternion [x, y, z, w] conversion
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize a vector. If length is near zero, returns fallback or raises ValueError."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        if fallback is None:
            raise ValueError("Cannot normalize a near-zero vector")
        return np.asarray(fallback, dtype=np.float64).reshape(-1)
    return v / norm


def rotation_aligning(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Shortest rotation matrix R such that R @ src = dst (twist-free swing)."""
    a = normalize(src, np.array([0.0, 1.0, 0.0]))
    b = normalize(dst, np.array([0.0, 1.0, 0.0]))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    sine = float(np.linalg.norm(axis))

    if cosine > 0.999999:
        return np.eye(3, dtype=np.float64)
    if cosine < -0.999999:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def orthonormal_frame(left: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed axes: columns = (left=+X, up=+Y, forward=+Z)."""
    up_u = normalize(up, np.array([0.0, 1.0, 0.0]))
    left_rej = left - up_u * float(np.dot(left, up_u))
    if float(np.linalg.norm(left_rej)) < 1e-8:
        helper = np.array([1.0, 0.0, 0.0]) if abs(up_u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        left_rej = np.cross(helper, up_u)
    left_u = normalize(left_rej)
    fwd = normalize(np.cross(left_u, up_u))
    left_u = normalize(np.cross(up_u, fwd))
    return np.column_stack([left_u, up_u, fwd])


def frame_from_primary_secondary(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Orthonormal basis: columns = (primary, secondary_ortho, primary x secondary_ortho)."""
    p = normalize(primary, np.array([0.0, 1.0, 0.0]))
    s = secondary - p * float(np.dot(secondary, p))
    if float(np.linalg.norm(s)) < 1e-8:
        helper = np.array([0.0, 1.0, 0.0]) if abs(p[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        s = np.cross(helper, p)
    s = normalize(s)
    t = np.cross(p, s)
    return np.column_stack([p, s, t])


def frame_anim_from_axes(
    r_parent_world: np.ndarray,
    rest_left_local: np.ndarray,
    rest_up_local: np.ndarray,
    tgt_left_world: np.ndarray,
    tgt_up_world: np.ndarray,
) -> np.ndarray:
    """Local rotation mapping rest local (left, up) onto world target axes.
    Used for Hips and Chest so clavicles maintain realistic torso twist.
    """
    r_rest_world = r_parent_world
    r_rest_frame = orthonormal_frame(r_rest_world @ rest_left_local, r_rest_world @ rest_up_local)
    r_tgt_frame = orthonormal_frame(tgt_left_world, tgt_up_world)
    r_world = (r_tgt_frame @ r_rest_frame.T) @ r_rest_world
    return r_parent_world.T @ r_world


def rotation_180_around(axis: np.ndarray) -> np.ndarray:
    """180-degree antipodal rotation matrix around a unit axis."""
    n = normalize(axis, np.array([0.0, 1.0, 0.0]))
    return 2.0 * np.outer(n, n) - np.eye(3, dtype=np.float64)


def geodesic_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic distance on SO(3) manifold in degrees between two rotation matrices."""
    rel = np.asarray(r_a, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64)
    cosine = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pick_closest_aim_twist(
    r_curr: np.ndarray,
    r_prev: Optional[np.ndarray],
    axis: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Preserve aim while resolving discrete 180° twist ambiguity against previous frame."""
    r_curr = np.asarray(r_curr, dtype=np.float64)
    if r_prev is None:
        return r_curr, False
    r_prev = np.asarray(r_prev, dtype=np.float64)
    r_flip = r_curr @ rotation_180_around(axis)
    if geodesic_angle_deg(r_flip, r_prev) + 1e-9 < geodesic_angle_deg(r_curr, r_prev):
        return r_flip, True
    return r_curr, False


def matrix_to_quat_xyzw(rmat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to glTF quaternion [x, y, z, w]."""
    m = np.asarray(rmat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0  # s = 4 * w
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0  # s = 4 * x
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0  # s = 4 * y
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0  # s = 4 * z
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert glTF quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def continue_quaternion_xyzw(q_curr: np.ndarray, q_prev: Optional[np.ndarray]) -> np.ndarray:
    """Enforce sign continuity between frames to eliminate antipodal popping in glTF."""
    if q_prev is None:
        return q_curr.copy()
    if float(np.dot(q_curr, q_prev)) < 0.0:
        return -q_curr
    return q_curr.copy()


# Mapping from MediaPipe world tracking space (left=+X, up=+Y, forward=+Z)
# to 3D model world space (left=+X, up=+Y, forward=+Z):
# Both are standardized right-handed Y-up coordinates.
R_BODY_ALIGN = np.eye(3, dtype=np.float64)


def human_to_canonical(p: np.ndarray) -> np.ndarray:
    """Map human joint coordinates (m) to canonical 3D model world space."""
    return np.asarray(p, dtype=np.float64) @ R_BODY_ALIGN.T

