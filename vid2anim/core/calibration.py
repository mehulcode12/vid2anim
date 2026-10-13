"""Rest Pose Calibration: T-Pose and A-Pose Alignment.

Different 3D characters arrive in different bind/rest poses:
- T-Pose: Arms horizontal (Mixamo standard).
- A-Pose: Arms angled downwards at ~30°-45° (VRM, Ready Player Me, Unreal Engine).

This module calibrates rest orientation offsets to prevent arm elevation distortions.
"""

from __future__ import annotations

from enum import Enum
import numpy as np

from ..utils.math_3d import quat_from_axis_angle, quat_identity, quat_multiply


class RestPoseType(str, Enum):
    T_POSE = "t_pose"
    A_POSE_30 = "a_pose_30"
    A_POSE_45 = "a_pose_45"
    CUSTOM = "custom"


class PoseCalibrator:
    """Computes rotational offsets between detected rest pose and canonical motion."""

    def __init__(self, rest_pose: RestPoseType = RestPoseType.T_POSE) -> None:
        self.rest_pose = rest_pose

    def get_rest_offset(self, canonical_bone: str) -> np.ndarray:
        """Returns the corrective offset quaternion [x, y, z, w] for the rest pose."""
        if self.rest_pose == RestPoseType.T_POSE:
            return quat_identity()

        # A-Pose adjustments: rotate shoulder/arm up to match T-pose reference frame
        angle_rad = np.radians(35.0) if self.rest_pose == RestPoseType.A_POSE_30 else np.radians(45.0)

        # For Left arm: A-pose rest is rotated downwards around Z axis
        if canonical_bone in ["left_arm", "left_shoulder"]:
            # Rotate back around forward axis (Z)
            return quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), angle_rad)
        elif canonical_bone in ["right_arm", "right_shoulder"]:
            return quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), -angle_rad)

        return quat_identity()
