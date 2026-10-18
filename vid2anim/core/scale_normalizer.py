"""Metric Scale and Trajectory Normalizer.

Ensures root motion stride matches character physical height across different unit
conventions (meters, centimeters, millimeters).
"""

from __future__ import annotations

import numpy as np

from .motion_extractor import MotionSequence
from .rig_detector import RigAnalysis


class ScaleNormalizer:
    """Normalizes displacement trajectories based on character proportion."""

    def __init__(self, target_height_m: float = 1.75) -> None:
        self.target_height_m = max(0.2, target_height_m)

    def estimate_model_scale(self, rig_info: RigAnalysis) -> float:
        """Estimate scaling multiplier to convert model units to standard meters."""
        # Standard default in GLTF is meters (scale = 1.0)
        # FBX files frequently use centimeters (scale = 0.01)
        if rig_info.format.upper() == "FBX":
            return 0.01
        return 1.0

    def scale_trajectory(
        self,
        motion: MotionSequence,
        character_scale: float = 1.0,
    ) -> MotionSequence:
        """Scale root motion translation to match character dimensions."""
        if character_scale == 1.0:
            return motion

        scaled_root = motion.root_translation * character_scale
        scaled_joints = {
            k: v * character_scale for k, v in motion.joint_positions.items()
        }

        return MotionSequence(
            fps=motion.fps,
            duration_sec=motion.duration_sec,
            frame_count=motion.frame_count,
            timestamps=motion.timestamps,
            joint_positions=scaled_joints,
            root_translation=scaled_root.astype(np.float32),
        )
