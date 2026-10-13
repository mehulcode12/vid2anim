"""Occlusion and Outlier Handling for 3D Pose Tracking.

Monocular tracking often produces sudden single-frame landmark dropouts (NaNs)
or extreme velocity spikes (e.g., from motion blur or rapid camera movement).
This module sanitizes coordinates, clamps velocity spikes, and fills missing frames.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from .motion_extractor import MotionSequence


def sanitize_and_interpolate_motion(
    motion: MotionSequence,
    max_velocity_mps: float = 8.0,
) -> MotionSequence:
    """Sanitize NaNs, clamp excessive teleportations, and interpolate tracking gaps.

    Args:
        motion: Raw extracted motion sequence.
        max_velocity_mps: Maximum reasonable joint speed in meters per second (default: 8.0 m/s).
    """
    num_frames = motion.frame_count
    if num_frames <= 1:
        return motion

    dt = 1.0 / motion.fps if motion.fps > 0 else 0.033333
    max_step_dist = max_velocity_mps * dt

    cleaned_joints: Dict[str, np.ndarray] = {}

    for j_name, pos_array in motion.joint_positions.items():
        # Safeguard: Ensure array length exactly matches num_frames
        if len(pos_array) != num_frames:
            if len(pos_array) == 0:
                pos = np.zeros((num_frames, 3), dtype=np.float32)
            elif len(pos_array) == 1:
                pos = np.repeat(pos_array, num_frames, axis=0).astype(np.float32)
            else:
                pos = np.zeros((num_frames, 3), dtype=np.float32)
                copy_len = min(len(pos_array), num_frames)
                pos[:copy_len] = pos_array[:copy_len]
                if copy_len < num_frames:
                    pos[copy_len:] = pos_array[-1]
        else:
            pos = pos_array.copy()

        # Step 1: Detect NaNs and Infs
        invalid_mask = np.isnan(pos).any(axis=1) | np.isinf(pos).any(axis=1)

        # Step 2: Detect velocity spikes
        for f in range(1, num_frames):
            if not invalid_mask[f] and not invalid_mask[f - 1]:
                delta = np.linalg.norm(pos[f] - pos[f - 1])
                if delta > max_step_dist:
                    invalid_mask[f] = True

        # Step 3: Interpolate invalid frames
        valid_indices = np.where(~invalid_mask)[0]
        if len(valid_indices) == 0:
            # If all are invalid, fallback to origin
            pos = np.zeros_like(pos)
        elif len(valid_indices) < num_frames:
            # Interpolate for each coordinate (X, Y, Z)
            for coord in range(3):
                pos[:, coord] = np.interp(
                    np.arange(num_frames),
                    valid_indices,
                    pos[valid_indices, coord],
                )

        cleaned_joints[j_name] = pos.astype(np.float32)

    # Sanitize root translation
    if len(motion.root_translation) != num_frames:
        if len(motion.root_translation) == 0:
            root = np.zeros((num_frames, 3), dtype=np.float32)
        elif len(motion.root_translation) == 1:
            root = np.repeat(motion.root_translation, num_frames, axis=0).astype(np.float32)
        else:
            root = np.zeros((num_frames, 3), dtype=np.float32)
            copy_len = min(len(motion.root_translation), num_frames)
            root[:copy_len] = motion.root_translation[:copy_len]
            if copy_len < num_frames:
                root[copy_len:] = motion.root_translation[-1]
    else:
        root = motion.root_translation.copy()

    root_invalid = np.isnan(root).any(axis=1) | np.isinf(root).any(axis=1)
    valid_root_idx = np.where(~root_invalid)[0]
    if 0 < len(valid_root_idx) < num_frames:
        for coord in range(3):
            root[:, coord] = np.interp(np.arange(num_frames), valid_root_idx, root[valid_root_idx, coord])
    elif len(valid_root_idx) == 0:
        root = np.zeros_like(root)

    return MotionSequence(
        fps=motion.fps,
        duration_sec=motion.duration_sec,
        frame_count=num_frames,
        timestamps=motion.timestamps,
        joint_positions=cleaned_joints,
        root_translation=root.astype(np.float32),
    )
