"""Universal Kinematic Retargeting Solver.

Converts 3D extracted motion into target bone rotations and root translations.
Uses vector-aiming swing-twist mathematics to transfer motion seamlessly
onto any humanoid armature (Mixamo, VRM, Rigify, Unreal, or Auto-Rigged meshes)
without requiring identical bone lengths or orientations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..utils.math_3d import continue_quaternion, normalize, quat_identity, quat_multiply, rotation_between_vectors
from ..utils.progress import ProgressReporter
from .calibration import PoseCalibrator
from .motion_extractor import MotionSequence
from .rig_detector import CANONICAL_JOINTS, RigAnalysis


# Joint pairs defining limb directional vectors
LIMB_VECTORS: Dict[str, Tuple[str, str, np.ndarray]] = {
    # canonical_bone: (parent_joint, child_joint, default_rest_direction)
    "spine": ("hips", "spine", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    "chest": ("spine", "chest", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    "neck": ("chest", "neck", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    "head": ("neck", "head", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    "left_shoulder": ("chest", "left_shoulder", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    "left_arm": ("left_arm", "left_forearm", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    "left_forearm": ("left_forearm", "left_hand", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    "right_shoulder": ("chest", "right_shoulder", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
    "right_arm": ("right_arm", "right_forearm", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
    "right_forearm": ("right_forearm", "right_hand", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
    "left_upleg": ("left_upleg", "left_leg", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
    "left_leg": ("left_leg", "left_foot", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
    "right_upleg": ("right_upleg", "right_leg", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
    "right_leg": ("right_leg", "right_foot", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
}


@dataclass
class RetargetedAnimation:
    """Baked animation tracks ready for 3D engine export."""
    fps: float
    duration_sec: float
    frame_count: int
    timestamps: np.ndarray  # (F,) in seconds
    bone_rotations: Dict[str, np.ndarray]  # bone_name -> (F, 4) quaternions [x, y, z, w]
    root_translations: np.ndarray  # (F, 3) translations [x, y, z]


class UniversalRetargeter:
    """Solves per-frame bone rotations for any mapped humanoid rig."""

    def __init__(self, reporter: Optional[ProgressReporter] = None) -> None:
        self.reporter = reporter or ProgressReporter()

    def retarget(
        self,
        motion: MotionSequence,
        rig: RigAnalysis,
        calibrator: Optional[PoseCalibrator] = None,
    ) -> RetargetedAnimation:
        """Retarget the extracted motion sequence onto the target rig."""
        num_frames = motion.frame_count
        mapping = rig.bone_mapping

        self.reporter.report(
            "RETARGET",
            0.1,
            f"Retargeting {num_frames} frames onto {rig.rig_type.value} rig ({len(mapping)} mapped bones)",
        )

        bone_rotations: Dict[str, np.ndarray] = {}
        for canonical, target_bone in mapping.items():
            bone_rotations[target_bone] = np.zeros((num_frames, 4), dtype=np.float32)

        prev_rotations: Dict[str, np.ndarray] = {
            target_bone: quat_identity() for target_bone in mapping.values()
        }

        # Solve per-frame rotations
        for f in range(num_frames):
            for canonical, target_bone in mapping.items():
                if canonical in LIMB_VECTORS:
                    p_joint, c_joint, rest_vec = LIMB_VECTORS[canonical]
                    
                    if p_joint in motion.joint_positions and c_joint in motion.joint_positions:
                        pos_parent = motion.joint_positions[p_joint][f]
                        pos_child = motion.joint_positions[c_joint][f]
                        curr_vec = pos_child - pos_parent

                        if np.linalg.norm(curr_vec) > 1e-5:
                            curr_dir = normalize(curr_vec)
                            q_rot = rotation_between_vectors(rest_vec, curr_dir)
                            
                            # Apply rest-pose corrective offset if calibrator provided
                            if calibrator is not None:
                                q_offset = calibrator.get_rest_offset(canonical)
                                q_rot = quat_multiply(q_rot, q_offset)

                            # Preserve sign continuity
                            q_clean = continue_quaternion(q_rot, prev_rotations[target_bone])
                            prev_rotations[target_bone] = q_clean
                            bone_rotations[target_bone][f] = q_clean
                        else:
                            bone_rotations[target_bone][f] = prev_rotations[target_bone]
                    else:
                        bone_rotations[target_bone][f] = quat_identity()
                else:
                    # Root or unconstrained bone
                    bone_rotations[target_bone][f] = quat_identity()

            if f % 30 == 0 or f == num_frames - 1:
                progress = 0.1 + 0.85 * ((f + 1) / num_frames)
                self.reporter.report("RETARGET", progress, f"Baked frame {f+1}/{num_frames}")

        self.reporter.report("RETARGET", 1.0, "Kinematic retargeting complete.")

        return RetargetedAnimation(
            fps=motion.fps,
            duration_sec=motion.duration_sec,
            frame_count=num_frames,
            timestamps=motion.timestamps,
            bone_rotations=bone_rotations,
            root_translations=motion.root_translation,
        )
