"""3D Vector and Quaternion Math for Skeletal Animation and Retargeting.

All quaternions follow the standard GLTF convention [x, y, z, w] unless explicitly stated.
Vector operations are vectorized using NumPy for fast performance on CPU.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a 3D vector or array of vectors along the last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > eps, v / norm, np.zeros_like(v))


def quat_identity() -> np.ndarray:
    """Return an identity quaternion [x=0, y=0, z=0, w=1]."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Compute quaternion [x, y, z, w] from a 3D axis and rotation angle in radians."""
    axis = normalize(np.asarray(axis, dtype=np.float32))
    half_angle = angle_rad * 0.5
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions: q_res = q1 * q2."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    res = np.stack([x, y, z, w], axis=-1)
    return normalize(res)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate/inverse of a unit quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector v by quaternion q."""
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    q_inv = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, qv), q_inv)
    return rotated[:3]


def rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Calculate the shortest swing rotation quaternion from v_from to v_to."""
    v_from = normalize(v_from)
    v_to = normalize(v_to)

    dot = np.dot(v_from, v_to)

    # Identical direction
    if dot >= 1.0 - 1e-7:
        return quat_identity()

    # Opposite direction (180 degree flip)
    if dot <= -1.0 + 1e-7:
        # Find any perpendicular axis
        ortho = np.cross(v_from, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(ortho) < 1e-4:
            ortho = np.cross(v_from, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return quat_from_axis_angle(ortho, np.pi)

    # Standard case
    axis = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
    return normalize(q)


def continue_quaternion(q_curr: np.ndarray, q_prev: np.ndarray | None) -> np.ndarray:
    """Ensure sign continuity between frames: q and -q represent the same 3D rotation,
    but sign flipping causes nasty interpolation pops.
    """
    if q_prev is None:
        return q_curr.copy()
    if np.dot(q_curr, q_prev) < 0.0:
        return -q_curr
    return q_curr.copy()


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear approximation for tiny angles
        res = q0 + t * (q1 - q0)
        return normalize(res)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return normalize(s0 * q0 + s1 * q1)


def decompose_swing_twist(q: np.ndarray, twist_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose quaternion q into swing and twist around a specific axis:
    q = q_swing * q_twist
    """
    twist_axis = normalize(twist_axis)
    # Project vector part of q onto twist axis
    q_vec = q[:3]
    proj = np.dot(q_vec, twist_axis) * twist_axis

    q_twist = np.array([proj[0], proj[1], proj[2], q[3]], dtype=np.float32)
    if np.linalg.norm(q_twist) < 1e-7:
        q_twist = quat_identity()
    else:
        q_twist = normalize(q_twist)

    # q_swing = q * q_twist^-1
    q_swing = quat_multiply(q, quat_conjugate(q_twist))
    return q_swing, q_twist
