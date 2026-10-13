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


LIMB_CHILD: Dict[str, str] = {
    "spine": "chest",
    "chest": "neck",
    "neck": "head",
    "left_shoulder": "left_arm",
    "left_arm": "left_forearm",
    "left_forearm": "left_hand",
    "right_shoulder": "right_arm",
    "right_arm": "right_forearm",
    "right_forearm": "right_hand",
    "left_upleg": "left_leg",
    "left_leg": "left_foot",
    "right_upleg": "right_leg",
    "right_leg": "right_foot",
}

LIMB_HUMAN_PAIRS: Dict[str, Tuple[str, str]] = {
    "spine": ("hips", "spine"),
    "chest": ("spine", "chest"),
    "neck": ("chest", "neck"),
    "head": ("neck", "head"),
    "left_shoulder": ("chest", "left_shoulder"),
    "left_arm": ("left_shoulder", "left_forearm"),
    "left_forearm": ("left_forearm", "left_hand"),
    "right_shoulder": ("chest", "right_shoulder"),
    "right_arm": ("right_shoulder", "right_forearm"),
    "right_forearm": ("right_forearm", "right_hand"),
    "left_upleg": ("left_upleg", "left_leg"),
    "left_leg": ("left_leg", "left_foot"),
    "right_upleg": ("right_upleg", "right_leg"),
    "right_leg": ("right_leg", "right_foot"),
}

SOLVE_ORDER: List[str] = [
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_arm",
    "left_forearm",
    "left_hand",
    "right_shoulder",
    "right_arm",
    "right_forearm",
    "right_hand",
    "left_upleg",
    "left_leg",
    "left_foot",
    "right_upleg",
    "right_leg",
    "right_foot",
]


class UniversalTransferSolver:
    """Computes exact per-bone basis change and transfers motion onto target glTF rigs."""

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

        # Precompute true topological local aim axes from rest child offsets
        self.aim_local: Dict[str, np.ndarray] = {}
        self._build_topological_aim_axes()

        # Compute basis change matrices for inspection and compatibility
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

    def _build_topological_aim_axes(self) -> None:
        """Discover each bone's physical heading vector in its local coordinate frame."""
        for canon_p, canon_c in LIMB_CHILD.items():
            if canon_p in self.rig_info.bone_mapping and canon_c in self.rig_info.bone_mapping:
                p_b = self.rig_info.bone_mapping[canon_p]
                c_b = self.rig_info.bone_mapping[canon_c]
                if p_b in self.name_to_idx and c_b in self.name_to_idx:
                    p_idx = self.name_to_idx[p_b]
                    c_idx = self.name_to_idx[c_b]
                    pos_p = self.rest_world_mat[p_idx][:3, 3]
                    pos_c = self.rest_world_mat[c_idx][:3, 3]
                    v_w = pos_c - pos_p
                    if float(np.linalg.norm(v_w)) > 1e-5:
                        dir_w = normalize(v_w)
                        m_p = self.rest_world_mat[p_idx][:3, :3]
                        m_p_rot = _orthonormalize_matrix(m_p)
                        self.aim_local[canon_p] = m_p_rot.T @ dir_w

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
                c = f_target.T @ f_skel
                self.basis_change[target_bone] = _orthonormalize_matrix(c)

    def transfer_motion(
        self,
        canonical_motion: CanonicalMotion,
        raw_motion: Optional[Any] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Transfer 3D kinematics onto target rig bones using topological aim and rest preservation.

        Returns:
            bone_rotations: target_bone_name -> (F, 4) quaternions [x, y, z, w]
            root_translations: (F, 3) translations [x, y, z] for hips
        """
        from .kinematics import orthonormal_frame, pick_closest_aim_twist, rotation_aligning

        num_frames = canonical_motion.frame_count
        bone_rotations: Dict[str, np.ndarray] = {}
        prev_quat: Dict[str, np.ndarray] = {}
        prev_rot_mat: Dict[str, np.ndarray] = {}

        # Resolve joint positions source
        joint_pos: Dict[str, np.ndarray] = {}
        if raw_motion is not None and hasattr(raw_motion, "joint_positions") and raw_motion.joint_positions:
            joint_pos = raw_motion.joint_positions
        elif canonical_motion.joint_positions is not None and len(canonical_motion.joint_positions) > 0:
            joint_pos = canonical_motion.joint_positions
        else:
            # Fallback tohips translation only if positions missing
            joint_pos = {}

        for canonical, target_bone in self.rig_info.bone_mapping.items():
            if target_bone in self.name_to_idx:
                bone_rotations[target_bone] = np.zeros((num_frames, 4), dtype=np.float32)
                idx = self.name_to_idx[target_bone]
                q_init = self.rest_local_quat[idx].astype(np.float32)
                prev_quat[target_bone] = q_init
                prev_rot_mat[target_bone] = quat_xyzw_to_matrix(q_init)

        # Baseline reference frames at frame 0 (neutral reference)
        r_pelvis_ref = np.eye(3, dtype=np.float64)
        r_chest_ref = np.eye(3, dtype=np.float64)

        if "left_upleg" in joint_pos and "right_upleg" in joint_pos and "neck" in joint_pos and "hips" in joint_pos:
            p_l_hip0 = joint_pos["left_upleg"][0]
            p_r_hip0 = joint_pos["right_upleg"][0]
            p_neck0 = joint_pos["neck"][0]
            p_hips0 = joint_pos["hips"][0]
            r_pelvis_ref = orthonormal_frame(p_l_hip0 - p_r_hip0, p_neck0 - p_hips0)

        if "left_shoulder" in joint_pos and "right_shoulder" in joint_pos and "neck" in joint_pos and "spine" in joint_pos:
            p_l_sh0 = joint_pos["left_shoulder"][0]
            p_r_sh0 = joint_pos["right_shoulder"][0]
            p_spine0 = joint_pos["spine"][0]
            p_neck0 = joint_pos["neck"][0]
            r_chest_ref = orthonormal_frame(p_l_sh0 - p_r_sh0, p_neck0 - p_spine0)

        # Solve per-frame hierarchical rotations
        for f in range(num_frames):
            curr_world_rot: Dict[int, np.ndarray] = {}
            for idx in range(len(self.nodes)):
                m_w = self.rest_world_mat[idx][:3, :3]
                curr_world_rot[idx] = _orthonormalize_matrix(m_w)

            for canon in SOLVE_ORDER:
                if canon not in self.rig_info.bone_mapping:
                    continue
                target_bone = self.rig_info.bone_mapping[canon]
                if target_bone not in self.name_to_idx:
                    continue
                idx = self.name_to_idx[target_bone]
                p_idx = self.parent_of.get(idx)
                w_parent = curr_world_rot.get(p_idx, np.eye(3, dtype=np.float64)) if p_idx is not None else np.eye(3, dtype=np.float64)
                r_bone_rest = quat_xyzw_to_matrix(self.rest_local_quat[idx])
                w_bone_rest = _orthonormalize_matrix(self.rest_world_mat[idx][:3, :3])

                r_local = r_bone_rest

                if canon == "hips":
                    if "left_upleg" in joint_pos and "right_upleg" in joint_pos and "neck" in joint_pos and "hips" in joint_pos:
                        v_lat = joint_pos["left_upleg"][f] - joint_pos["right_upleg"][f]
                        v_up = joint_pos["neck"][f] - joint_pos["hips"][f]
                        r_pelvis = orthonormal_frame(v_lat, v_up)
                        delta_pelvis = r_pelvis @ r_pelvis_ref.T
                        r_w = delta_pelvis @ w_bone_rest
                        r_local = w_parent.T @ r_w
                    elif f < len(canonical_motion.r_anim):
                        r_canon_f = canonical_motion.r_anim[f].get("hips", np.eye(3))
                        r_local = r_bone_rest @ r_canon_f

                elif canon == "chest":
                    if "left_shoulder" in joint_pos and "right_shoulder" in joint_pos and "neck" in joint_pos and "spine" in joint_pos:
                        v_lat = joint_pos["left_shoulder"][f] - joint_pos["right_shoulder"][f]
                        v_up = joint_pos["neck"][f] - joint_pos["spine"][f]
                        r_chest = orthonormal_frame(v_lat, v_up)
                        delta_chest = r_chest @ r_chest_ref.T
                        r_w = delta_chest @ w_bone_rest
                        r_local = w_parent.T @ r_w
                    elif f < len(canonical_motion.r_anim):
                        r_canon_f = canonical_motion.r_anim[f].get("chest", np.eye(3))
                        r_local = r_bone_rest @ r_canon_f

                elif canon in self.aim_local and canon in LIMB_HUMAN_PAIRS:
                    d_aim = self.aim_local[canon]
                    hp_p, hp_c = LIMB_HUMAN_PAIRS[canon]
                    if hp_p in joint_pos and hp_c in joint_pos:
                        pos_p = joint_pos[hp_p][f]
                        pos_c = joint_pos[hp_c][f]
                        v_human = pos_c - pos_p
                        if float(np.linalg.norm(v_human)) > 1e-5:
                            target_w = normalize(v_human)
                            target_in_parent = w_parent.T @ target_w
                            r_delta = rotation_aligning(r_bone_rest @ d_aim, target_in_parent)
                            r_local = r_delta @ r_bone_rest
                        else:
                            r_local = r_bone_rest
                    else:
                        r_local = r_bone_rest

                elif f < len(canonical_motion.r_anim):
                    # For joints without child limbs, transfer basis delta from canonical
                    d = canonical_motion.r_anim[f].get(canon, np.eye(3, dtype=np.float64))
                    c_mat = self.basis_change.get(target_bone, np.eye(3, dtype=np.float64))
                    d_prime = c_mat @ d @ c_mat.T
                    r_local = r_bone_rest @ d_prime

                # Geodesic anti-twist smoothing
                aim_axis = self.aim_local.get(canon, np.array([0.0, 1.0, 0.0], dtype=np.float64))
                r_clean, _ = pick_closest_aim_twist(r_local, prev_rot_mat.get(target_bone), aim_axis)
                prev_rot_mat[target_bone] = r_clean

                # Update world rotation tracking for downstream children
                curr_world_rot[idx] = w_parent @ r_clean

                # Quaternion continuity
                q_anim = matrix_to_quat_xyzw(r_clean)
                q_clean = continue_quaternion_xyzw(q_anim, prev_quat.get(target_bone))
                prev_quat[target_bone] = q_clean
                bone_rotations[target_bone][f] = q_clean

        # Root translation for hips: relative delta from frame 0
        root_bone = self.rig_info.bone_mapping.get("hips")
        root_trans = np.zeros((num_frames, 3), dtype=np.float32)
        if root_bone and root_bone in self.name_to_idx:
            if "hips" in joint_pos:
                hips_p = joint_pos["hips"].astype(np.float32)
                root_trans = hips_p - hips_p[0:1]
            else:
                canon_trans = canonical_motion.hips_translations.astype(np.float32)
                root_trans = canon_trans - canon_trans[0:1]

        return bone_rotations, root_trans

