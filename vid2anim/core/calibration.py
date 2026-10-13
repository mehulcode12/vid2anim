"""Rest Pose Calibration: T-Pose and A-Pose Alignment.

Different 3D characters arrive in different bind/rest poses:
- T-Pose: Arms horizontal (Mixamo standard).
- A-Pose: Arms angled downwards at ~30°-45° (VRM, Ready Player Me, Unreal Engine).

This module calibrates rest orientation offsets to prevent arm elevation distortions.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np

from ..utils.math_3d import quat_from_axis_angle, quat_identity, quat_multiply


class RestPoseType(str, Enum):
    AUTO = "auto"
    T_POSE = "t_pose"
    A_POSE_30 = "a_pose_30"
    A_POSE_45 = "a_pose_45"
    CUSTOM = "custom"


class PoseCalibrator:
    """Computes rotational offsets between detected rest pose and canonical motion."""

    def __init__(
        self,
        rest_pose: RestPoseType = RestPoseType.AUTO,
        arm_angle_deg: Optional[float] = None,
    ) -> None:
        self.rest_pose = rest_pose
        self.arm_angle_deg = arm_angle_deg

    @staticmethod
    def calculate_rest_arm_angle(
        arm_pos: np.ndarray,
        forearm_pos: np.ndarray,
        vertical_axis: int = 1,
    ) -> float:
        """Calculate the downward inclination angle (in degrees) of the upper arm in rest pose.
        
        Args:
            arm_pos: [x, y, z] coordinate of shoulder/upperarm
            forearm_pos: [x, y, z] coordinate of forearm/elbow
            vertical_axis: Index of the vertical axis (1 for Y-up, 2 for Z-up)
        """
        vec = forearm_pos - arm_pos
        lateral = abs(float(vec[0]))
        vert = float(vec[vertical_axis])

        # If arm is horizontal, vert is ~0, lateral is positive -> angle ~ 0 (T-Pose)
        # If arm is angled downwards, -vert > 0 -> angle > 0 (A-Pose)
        if lateral < 1e-4:
            return 90.0 if vert < 0 else 0.0

        angle_rad = np.arctan2(max(0.0, -vert), lateral)
        return float(np.degrees(angle_rad))

    def get_rest_offset(self, canonical_bone: str) -> np.ndarray:
        """Returns the corrective offset quaternion [x, y, z, w] for the rest pose."""
        if self.rest_pose == RestPoseType.T_POSE:
            return quat_identity()

        # Determine effective angle
        if self.rest_pose == RestPoseType.AUTO:
            if self.arm_angle_deg is None or self.arm_angle_deg < 10.0:
                return quat_identity()
            angle_rad = np.radians(self.arm_angle_deg)
        elif self.rest_pose == RestPoseType.A_POSE_30:
            angle_rad = np.radians(30.0)
        elif self.rest_pose == RestPoseType.A_POSE_45:
            angle_rad = np.radians(45.0)
        elif self.arm_angle_deg is not None:
            angle_rad = np.radians(self.arm_angle_deg)
        else:
            return quat_identity()

        # Rotate shoulder/arm up around Z (forward axis in Y-up convention)
        if canonical_bone in ["left_arm", "left_shoulder"]:
            return quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), float(angle_rad))
        elif canonical_bone in ["right_arm", "right_shoulder"]:
            return quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), float(-angle_rad))

        return quat_identity()
