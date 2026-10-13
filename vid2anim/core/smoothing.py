"""Temporal Smoothing and Jitter Reduction Filter for 3D Motion Capture.

Monocular computer-vision pose tracking produces high-frequency noise and jitter.
This module applies an adaptive One-Euro / dynamic exponential moving-average filter
for 3D joint positions and spherical interpolation (SLERP) for bone quaternions.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from ..utils.math_3d import continue_quaternion, normalize, quat_slerp
from .motion_extractor import MotionSequence
from .retargeter import RetargetedAnimation


class OneEuroFilter1D:
    """Adaptive low-pass filter: high smoothing when slow, minimal lag when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Derivative
        dx = (x - self.x_prev) / dt if dt > 0 else 0.0
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class TemporalMotionSmoother:
    """Smooths raw 3D positions and quaternions across video frames."""

    def __init__(self, position_smoothness: float = 1.0, rotation_window: int = 3) -> None:
        self.position_smoothness = max(0.1, position_smoothness)
        self.rotation_window = max(1, rotation_window)

    def smooth_motion_positions(self, motion: MotionSequence) -> MotionSequence:
        """Smooth 3D joint trajectories using adaptive filtering."""
        num_frames = motion.frame_count
        if num_frames <= 2:
            return motion

        dt = 1.0 / motion.fps if motion.fps > 0 else 0.033333
        smoothed_joints: Dict[str, np.ndarray] = {}

        for j_name, pos_array in motion.joint_positions.items():
            smoothed_pos = np.zeros_like(pos_array)
            # Create a 1D filter per XYZ coordinate
            fx = OneEuroFilter1D(min_cutoff=1.2 / self.position_smoothness, beta=0.03)
            fy = OneEuroFilter1D(min_cutoff=1.2 / self.position_smoothness, beta=0.03)
            fz = OneEuroFilter1D(min_cutoff=1.2 / self.position_smoothness, beta=0.03)

            for f in range(num_frames):
                smoothed_pos[f, 0] = fx.filter(float(pos_array[f, 0]), dt)
                smoothed_pos[f, 1] = fy.filter(float(pos_array[f, 1]), dt)
                smoothed_pos[f, 2] = fz.filter(float(pos_array[f, 2]), dt)

            smoothed_joints[j_name] = smoothed_pos

        # Smooth root translation
        smoothed_root = np.zeros_like(motion.root_translation)
        rx = OneEuroFilter1D(min_cutoff=0.8, beta=0.05)
        ry = OneEuroFilter1D(min_cutoff=0.8, beta=0.05)
        rz = OneEuroFilter1D(min_cutoff=0.8, beta=0.05)
        for f in range(num_frames):
            smoothed_root[f, 0] = rx.filter(float(motion.root_translation[f, 0]), dt)
            smoothed_root[f, 1] = ry.filter(float(motion.root_translation[f, 1]), dt)
            smoothed_root[f, 2] = rz.filter(float(motion.root_translation[f, 2]), dt)

        return MotionSequence(
            fps=motion.fps,
            duration_sec=motion.duration_sec,
            frame_count=num_frames,
            timestamps=motion.timestamps,
            joint_positions=smoothed_joints,
            root_translation=smoothed_root,
        )

    def smooth_animation_rotations(self, anim: RetargetedAnimation) -> RetargetedAnimation:
        """Smooth bone rotation quaternions across frames using SLERP windowing."""
        num_frames = anim.frame_count
        if num_frames <= 3 or self.rotation_window <= 1:
            return anim

        smoothed_rotations: Dict[str, np.ndarray] = {}

        for bone_name, quat_frames in anim.bone_rotations.items():
            smoothed_quats = np.zeros_like(quat_frames)
            smoothed_quats[0] = quat_frames[0]

            for f in range(1, num_frames - 1):
                q_prev = smoothed_quats[f - 1]
                q_curr = continue_quaternion(quat_frames[f], q_prev)
                q_next = continue_quaternion(quat_frames[f + 1], q_curr)

                # Blend previous and current
                q_blend1 = quat_slerp(q_prev, q_curr, 0.6)
                # Blend with forward frame
                q_final = quat_slerp(q_blend1, q_next, 0.3)
                smoothed_quats[f] = normalize(q_final)

            smoothed_quats[-1] = continue_quaternion(quat_frames[-1], smoothed_quats[-2])
            smoothed_rotations[bone_name] = smoothed_quats

        return RetargetedAnimation(
            fps=anim.fps,
            duration_sec=anim.duration_sec,
            frame_count=num_frames,
            timestamps=anim.timestamps,
            bone_rotations=smoothed_rotations,
            root_translations=anim.root_translations,
        )
