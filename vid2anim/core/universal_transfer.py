"""Universal Rig Transfer: C D C^T Basis Change Solver.

Ported from Motion2MixamoRig / GVHMR reference implementation.
Transforms canonical SO(3) rotations into the exact local bone coordinate frames
of ANY target glTF armature (Mixamo, Character Creator, Rigify, VRM, UE5, Daz3D, Sketchfab)
using the closed-form mathematical basis transformation:

    F'_target D' F'_target^T = R_align (F_skel D F_skel^T) R_align^T
    ==> D' = C D C^T,  where C = F'_target^T @ R_align @ F_skel
    ==> R_local_animated = R_target_rest_local @ D'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .canonical_skeleton import (
    CANONICAL_REST_WORLD_FRAME,
    CanonicalMotion,
)
from .kinematics import (
    continue_quaternion_xyzw,
    matrix_to_quat_xyzw,
    normalize,
    quat_xyzw_to_matrix,
)
from .rig_detector import RigAnalysis


def _orthonormalize_matrix(m: np.ndarray) -> np.ndarray:
    """Ensure a 3x3 matrix is strictly orthonormal with det=+1 using SVD."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if float(np.linalg.det(r)) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        r = u @ vt
    return r


class UniversalTransferSolver:
    """Computes exact per-bone basis change C and transfers motion onto target glTF rigs."""

    def __init__(self, gltf_data: Dict[str, Any], rig_info: RigAnalysis) -> None:
        self.gltf_data = gltf_data
        self.rig_info = rig_info
        self.nodes = gltf_data.get("nodes", [])
        
        # Build node indices, parents, and rest local transforms
        self.name_to_idx: Dict[str, int] = {}
        self.idx_to_name: Dict[int, str] = {}
        self.parent_of: Dict[int, Optional[int]] = {i: None for i in range(len(self.nodes))}
        self.rest_local_mat: Dict[int, np.ndarray] = {}
        self.rest_local_quat: Dict[int, np.ndarray] = {}
        self.rest_local_trans: Dict[int, np.ndarray] = {}

        for idx, node in enumerate(self.nodes):
            name = node.get("name", f"node_{idx}")
            self.name_to_idx[name] = idx
            self.idx_to_name[idx] = name

            for child_idx in node.get("children", []):
                self.parent_of[child_idx] = idx

            # Local transform components
            t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            q = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
            s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)

            self.rest_local_trans[idx] = t
            self.rest_local_quat[idx] = q

            r_mat = quat_xyzw_to_matrix(q)
            m = np.eye(4, dtype=np.float64)
            m[:3, :3] = r_mat @ np.diag(s)
            m[:3, 3] = t
            self.rest_local_mat[idx] = m

        # Compute rest world transform for all nodes
        self.rest_world_mat: Dict[int, np.ndarray] = {}
        for idx in range(len(self.nodes)):
            self._compute_world_mat(idx)

        # Compute F_target and C for each mapped bone
        self.basis_change: Dict[str, np.ndarray] = {}
        self._build_basis_changes()

    def _compute_world_mat(self, idx: int) -> np.ndarray:
        if idx in self.rest_world_mat:
            return self.rest_world_mat[idx]

        parent_idx = self.parent_of.get(idx)
        if parent_idx is None:
            w = self.rest_local_mat[idx].copy()
        else:
            w = self._compute_world_mat(parent_idx) @ self.rest_local_mat[idx]

        self.rest_world_mat[idx] = w
        return w

    def _build_basis_changes(self) -> None:
        """Derive basis change C = F_target^T @ F_skel for all mapped canonical bones."""
        for canonical, target_bone in self.rig_info.bone_mapping.items():
            if target_bone not in self.name_to_idx:
                continue
            idx = self.name_to_idx[target_bone]
            m_world = self.rest_world_mat[idx]
            f_target = _orthonormalize_matrix(m_world[:3, :3])

            if canonical in CANONICAL_REST_WORLD_FRAME:
                f_skel = CANONICAL_REST_WORLD_FRAME[canonical]
                # C = F_target^T @ F_skel
                c = f_target.T @ f_skel
                self.basis_change[target_bone] = _orthonormalize_matrix(c)

    def transfer_motion(
        self,
        canonical_motion: CanonicalMotion,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Transfer canonical SO(3) rotations onto target rig bone tracks.
        
        Returns:
            bone_rotations: target_bone_name -> (F, 4) quaternions [x, y, z, w]
            root_translations: (F, 3) translations [x, y, z] for hips
        """
        num_frames = canonical_motion.frame_count
        bone_rotations: Dict[str, np.ndarray] = {}
        prev_quat: Dict[str, np.ndarray] = {}

        # Reverse mapping: target_bone -> canonical
        target_to_canonical = {t: c for c, t in self.rig_info.bone_mapping.items()}

        for target_bone in self.basis_change:
            bone_rotations[target_bone] = np.zeros((num_frames, 4), dtype=np.float32)
            idx = self.name_to_idx[target_bone]
            prev_quat[target_bone] = self.rest_local_quat[idx].astype(np.float32)

        # Transfer each frame
        for f in range(num_frames):
            r_canonical_frame = canonical_motion.r_anim[f]

            for target_bone, c_mat in self.basis_change.items():
                canonical_joint = target_to_canonical[target_bone]
                d = r_canonical_frame.get(canonical_joint, np.eye(3, dtype=np.float64))

                # D' = C @ D @ C^T
                d_prime = c_mat @ d @ c_mat.T

                # R_local_animated = R_target_rest_local @ D'
                idx = self.name_to_idx[target_bone]
                r_rest_local = quat_xyzw_to_matrix(self.rest_local_quat[idx])
                r_animated = r_rest_local @ d_prime

                q_anim = matrix_to_quat_xyzw(r_animated)
                q_clean = continue_quaternion_xyzw(q_anim, prev_quat[target_bone])
                prev_quat[target_bone] = q_clean
                bone_rotations[target_bone][f] = q_clean

        # Root translation for hips: relative delta from frame 0
        root_bone = self.rig_info.bone_mapping.get("hips")
        root_trans = np.zeros((num_frames, 3), dtype=np.float32)
        if root_bone and root_bone in self.name_to_idx:
            canon_trans = canonical_motion.hips_translations.astype(np.float32)
            # Delta offset relative to frame 0
            root_trans = canon_trans - canon_trans[0:1]
        else:
            root_trans = canonical_motion.hips_translations.astype(np.float32)

        return bone_rotations, root_trans
