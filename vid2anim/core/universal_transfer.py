"""Universal Rig Transfer: Anatomical Reference & Closed-Form Forward Kinematics.

Ported and generalized from Motion2MixamoRig (MaxLiu22/Motion2MixamoRig) & GVHMR.
Transforms 3D human pose kinematics onto ANY target glTF armature (Mixamo,
Character Creator, Rigify, VRM, UE5, Daz3D, Sketchfab) using:
1. Strict rest-frame reference calculation from the 3D model's rest geometry (no frame-0 bias)
2. Exact two-axis orthonormal frame mapping for Hips and Torso/Chest
3. Step-by-step topological Forward Kinematics (FK) updating all ancestor and helper nodes
4. Foot ground-normal alignment and step aiming
5. Hand and wrist limb orientation
6. Contact-grounded vertical hip translation and metric skeleton scaling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .canonical_skeleton import (
    CANONICAL_REST_WORLD_FRAME,
    CanonicalMotion,
)
from .kinematics import (
    continue_quaternion_xyzw,
    frame_from_primary_secondary,
    matrix_to_quat_xyzw,
    normalize,
    orthonormal_frame,
    pick_closest_aim_twist,
    quat_xyzw_to_matrix,
    rotation_aligning,
)
from .rig_detector import RigAnalysis


def _orthonormalize_matrix(m: np.ndarray) -> np.ndarray:
    """Ensure a 3x3 matrix is strictly orthonormal with det=+1 using polar SVD."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if float(np.linalg.det(r)) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        r = u @ vt
    return r


LIMB_CHILD: Dict[str, str] = {
    "spine": "spine1",
    "spine1": "chest",
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
    "left_foot": "left_toe",
    "right_upleg": "right_leg",
    "right_leg": "right_foot",
    "right_foot": "right_toe",
}

LIMB_HUMAN_PAIRS: Dict[str, Tuple[str, str]] = {
    "spine": ("hips", "spine"),
    "spine1": ("spine", "spine1"),
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
    "left_foot": ("left_foot", "left_toe"),
    "right_upleg": ("right_upleg", "right_leg"),
    "right_leg": ("right_leg", "right_foot"),
    "right_foot": ("right_foot", "right_toe"),
}

SOLVE_ORDER: List[str] = [
    "hips",
    "spine",
    "spine1",
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
    "left_toe",
    "right_upleg",
    "right_leg",
    "right_foot",
    "right_toe",
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
        self.children_of: Dict[int, List[int]] = {i: [] for i in range(len(self.nodes))}
        self.rest_local_mat: Dict[int, np.ndarray] = {}
        self.rest_local_quat: Dict[int, np.ndarray] = {}
        self.rest_local_trans: Dict[int, np.ndarray] = {}

        for idx, node in enumerate(self.nodes):
            name = node.get("name", f"node_{idx}")
            self.name_to_idx[name] = idx
            self.idx_to_name[idx] = name

            for child_idx in node.get("children", []):
                self.parent_of[child_idx] = idx
                self.children_of[idx].append(child_idx)

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

        # Check if model has a pre-existing rest/pose animation (e.g. 'pose', 'rest', 'idle')
        # that defines preferred stowed/folded transforms for accessory bones (wings, swords, tail, boosters).
        self.accessory_baseline_rot: Dict[int, np.ndarray] = {}
        self.accessory_baseline_trans: Dict[int, np.ndarray] = {}
        model_path = getattr(self.rig_info, "file_path", None)
        if model_path and Path(model_path).exists() and gltf_data.get("animations"):
            try:
                import struct
                with open(model_path, "rb") as f_glb:
                    f_glb.seek(12)
                    j_len = int.from_bytes(f_glb.read(4), "little")
                    f_glb.seek(12 + 8 + j_len)
                    b_len = int.from_bytes(f_glb.read(4), "little")
                    f_glb.read(4)
                    bin_data = f_glb.read(b_len)

                anims = gltf_data.get("animations", [])
                preferred_anim = next(
                    (a for a in anims if any(p in a.get("name", "").lower() for p in ["pose", "rest", "idle", "default"])),
                    None,
                )
                if preferred_anim:
                    mapped_target_names = set(self.rig_info.bone_mapping.values())
                    for ch in preferred_anim.get("channels", []):
                        n_idx = ch["target"].get("node")
                        path_str = ch["target"].get("path")
                        if n_idx is None or n_idx not in self.idx_to_name:
                            continue
                        name_node = self.idx_to_name[n_idx]
                        if name_node in mapped_target_names:
                            continue
                        s_idx = ch["sampler"]
                        out_acc_idx = preferred_anim["samplers"][s_idx]["output"]
                        acc = gltf_data["accessors"][out_acc_idx]
                        bv = gltf_data["bufferViews"][acc["bufferView"]]
                        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                        if path_str == "rotation":
                            q_val = struct.unpack("<4f", bin_data[offset : offset + 16])
                            self.accessory_baseline_rot[n_idx] = np.array(q_val, dtype=np.float64)
                        elif path_str == "translation":
                            t_val = struct.unpack("<3f", bin_data[offset : offset + 12])
                            self.accessory_baseline_trans[n_idx] = np.array(t_val, dtype=np.float64)
            except Exception:
                pass

        # Apply accessory baseline transforms into rest_local_quat and rest_local_trans
        for acc_idx, acc_q in self.accessory_baseline_rot.items():
            self.rest_local_quat[acc_idx] = acc_q
            r_mat = quat_xyzw_to_matrix(acc_q)
            self.rest_local_mat[acc_idx][:3, :3] = r_mat
        for acc_idx, acc_t in self.accessory_baseline_trans.items():
            self.rest_local_trans[acc_idx] = acc_t
            self.rest_local_mat[acc_idx][:3, 3] = acc_t

        # Build topological sort of node hierarchy (parents strictly precede children)
        self.topological_order: List[int] = self._build_topological_order()

        # Compute rest world transform for all nodes in topological order
        self.rest_world_mat: Dict[int, np.ndarray] = {}
        for idx in self.topological_order:
            parent_idx = self.parent_of.get(idx)
            if parent_idx is None:
                self.rest_world_mat[idx] = self.rest_local_mat[idx].copy()
            else:
                self.rest_world_mat[idx] = self.rest_world_mat[parent_idx] @ self.rest_local_mat[idx]

        # Compute character rest height and unit scaling multiplier
        y_vals = [self.rest_world_mat[i][1, 3] for i in range(len(self.nodes))]
        self.model_height = float(max(y_vals) - min(y_vals)) if y_vals else 1.75
        self.units_per_meter = float(max(1.0, self.model_height / 1.75)) if self.model_height > 10.0 else 1.0

        # Cache rest world positions for mapped canonical joints
        self.rest_world_pos: Dict[str, np.ndarray] = {}
        for c, tb in self.rig_info.bone_mapping.items():
            if tb in self.name_to_idx:
                idx = self.name_to_idx[tb]
                self.rest_world_pos[c] = self.rest_world_mat[idx][:3, 3].copy()

        # Precompute true topological local aim axes from rest child offsets
        self.aim_local: Dict[str, np.ndarray] = {}
        self._build_topological_aim_axes()

        # Exact rest reference frames (from Motion2MixamoRig reference)
        # Hips rest frame from model's actual bone rest positions
        if "left_upleg" in self.rest_world_pos and "right_upleg" in self.rest_world_pos and "hips" in self.rest_world_pos:
            pos_l = self.rest_world_pos["left_upleg"]
            pos_r = self.rest_world_pos["right_upleg"]
            pos_h = self.rest_world_pos["hips"]
            pos_up = self.rest_world_pos.get(
                "neck",
                self.rest_world_pos.get("chest", self.rest_world_pos.get("spine", pos_h + np.array([0.0, 1.0, 0.0]))),
            )
            self.r_pelvis_rest = orthonormal_frame(pos_l - pos_r, pos_up - pos_h)
        else:
            self.r_pelvis_rest = np.eye(3, dtype=np.float64)

        # Chest rest frame from model's shoulders and spine/neck up
        if "left_shoulder" in self.rest_world_pos and "right_shoulder" in self.rest_world_pos:
            pos_l_sh = self.rest_world_pos["left_shoulder"]
            pos_r_sh = self.rest_world_pos["right_shoulder"]
            pos_spine = self.rest_world_pos.get("spine", self.rest_world_pos.get("chest", np.array([0.0, 1.0, 0.0])))
            pos_neck = self.rest_world_pos.get("neck", pos_spine + np.array([0.0, 1.0, 0.0]))
            self.r_chest_rest = orthonormal_frame(pos_l_sh - pos_r_sh, pos_neck - pos_spine)
        else:
            self.r_chest_rest = np.eye(3, dtype=np.float64)

        # Compute basis change matrices for compatibility
        self.basis_change: Dict[str, np.ndarray] = {}
        self._build_basis_changes()

        # Discover any foot bones detached from their lowerleg chain (e.g. Rigify/Blender FootIK)
        self.detached_foot_nodes: Dict[str, Tuple[int, int, int]] = {}
        for canon_foot, canon_leg in [("left_foot", "left_leg"), ("right_foot", "right_leg")]:
            if canon_foot in self.rig_info.bone_mapping and canon_leg in self.rig_info.bone_mapping:
                f_name = self.rig_info.bone_mapping[canon_foot]
                l_name = self.rig_info.bone_mapping[canon_leg]
                if f_name in self.name_to_idx and l_name in self.name_to_idx:
                    f_idx = self.name_to_idx[f_name]
                    l_idx = self.name_to_idx[l_name]
                    ancestors: List[int] = []
                    curr = self.parent_of.get(f_idx)
                    while curr is not None:
                        ancestors.append(curr)
                        curr = self.parent_of.get(curr)
                    if l_idx not in ancestors and self.parent_of.get(f_idx) is not None:
                        ik_node_idx = self.parent_of[f_idx]
                        self.detached_foot_nodes[canon_foot] = (f_idx, ik_node_idx, l_idx)

        # Discover procedural twist helpers and share bones
        self.twist_helpers: Dict[int, Tuple[int, float, str]] = {}
        self.share_bones: Dict[int, Tuple[int, float, str]] = {}
        mapped_target_names = set(self.rig_info.bone_mapping.values())

        for idx, node in enumerate(self.nodes):
            name = node.get("name", "")
            if name in mapped_target_names or "scalecompensation" in name.lower():
                continue
            name_l = name.lower()

            if "twist" in name_l or ("roll" in name_l and "ctrl" not in name_l):
                side_str = "left" if any(s in name_l for s in ["_l", ".l", "left", "l_"]) else "right"
                if any(k in name_l for k in ["forearm", "lowerarm", "elbow"]):
                    p_canon = f"{side_str}_forearm"
                    weight = 0.50
                elif any(k in name_l for k in ["shoulder", "clav", "collar", "shldr", "deltoid"]):
                    p_canon = f"{side_str}_shoulder"
                    weight = 0.50
                elif any(k in name_l for k in ["upperarm", "arm"]):
                    p_canon = f"{side_str}_arm"
                    weight = 0.50
                elif any(k in name_l for k in ["calf", "shin", "lowerleg", "knee"]):
                    p_canon = f"{side_str}_leg"
                    weight = 0.50
                elif any(k in name_l for k in ["thigh", "upperleg"]):
                    p_canon = f"{side_str}_upleg"
                    weight = 0.50
                elif "neck" in name_l:
                    p_canon = "neck"
                    weight = 0.50
                else:
                    p_canon = None
                    weight = 0.0

                if p_canon and p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.twist_helpers[idx] = (p_idx, weight, p_canon)

            elif "sharebone" in name_l or "share_bone" in name_l or "share" in name_l:
                side_str = "left" if any(s in name_l for s in ["_l", ".l", "left", "l_"]) else "right"
                if any(k in name_l for k in ["shoulder", "clav", "collar", "shldr", "deltoid"]):
                    p_canon = f"{side_str}_shoulder"
                    weight = 0.50
                elif "elbow" in name_l:
                    p_canon = f"{side_str}_forearm"
                    weight = -0.35
                elif "knee" in name_l:
                    p_canon = f"{side_str}_leg"
                    weight = -0.35
                elif "toe" in name_l:
                    p_canon = f"{side_str}_toe" if f"{side_str}_toe" in self.rig_info.bone_mapping else f"{side_str}_foot"
                    weight = 0.50
                else:
                    p_canon = None
                    weight = 0.0

                if p_canon and p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

            elif any(k in name_l for k in ["butt", "glute", "ass", "buttock", "pelvis_"]):
                side_str = "left" if any(s in name_l for s in ["_l", ".l", "left", "l_"]) else "right"
                p_canon = f"{side_str}_upleg"
                weight = 0.35
                if p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

            elif any(k in name_l for k in ["legbooster", "regbooster", "thighwing", "hipwing", "hipbooster"]):
                side_str = "left" if any(s in name_l for s in ["_l", ".l", "left", "l_"]) else "right"
                p_canon = f"{side_str}_upleg"
                weight = 0.20
                if p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

            elif any(k in name_l for k in ["tail_mount", "boosterpod"]):
                p_canon = "hips"
                weight = 0.25
                if p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

            elif "tail" in name_l:
                p_canon = "hips"
                weight = 0.30
                if p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

            elif "pelvis" in name_l and idx != self.name_to_idx.get(self.rig_info.bone_mapping.get("hips")):
                p_canon = "hips"
                weight = 0.50
                if p_canon in self.rig_info.bone_mapping:
                    primary_name = self.rig_info.bone_mapping[p_canon]
                    p_idx = self.name_to_idx.get(primary_name)
                    if p_idx is not None:
                        self.share_bones[idx] = (p_idx, weight, p_canon)

        # Discover finger bones for procedural relaxed posing & dynamic flexion
        import re
        self.finger_bones: List[Tuple[int, str, int, bool]] = []
        for idx, node in enumerate(self.nodes):
            name = node.get("name", "")
            if name in mapped_target_names or "scalecompensation" in name.lower() or "ctrl" in name.lower():
                continue
            name_l = name.lower()
            if any(k in name_l for k in ["thumb", "index", "mid", "middle", "ring", "pinky", "little", "rittre", "finger"]) and "toe" not in name_l and "share" not in name_l:
                side = "left" if any(s in name_l for s in ["_l", ".l", "left", "l_"]) else "right"
                m_ph = re.search(r'(?:thumb|index|mid|middle|ring|pinky|little|rittre)[^\d]*([123])', name_l)
                if m_ph:
                    phalanx = int(m_ph.group(1))
                elif any(p in name_l for p in ["distal", "03"]):
                    phalanx = 3
                elif any(p in name_l for p in ["inter", "02"]):
                    phalanx = 2
                else:
                    phalanx = 1
                is_thumb = "thumb" in name_l
                self.finger_bones.append((idx, side, phalanx, is_thumb))

        # Discover detached waist / hip accessory nodes (e.g. swords, scabbards, holsters, pouches, skirts)
        # that are siblings of the hips (children of hips' parent node)
        self.detached_hip_accessories: List[int] = []
        root_bone = self.rig_info.bone_mapping.get("hips")
        if root_bone and root_bone in self.name_to_idx:
            root_idx = self.name_to_idx[root_bone]
            root_parent = self.parent_of.get(root_idx)
            if root_parent is not None:
                for child_idx in self.children_of.get(root_parent, []):
                    if child_idx == root_idx:
                        continue
                    ch_name_l = self.idx_to_name[child_idx].lower()
                    if any(k in ch_name_l for k in ["sword", "blade", "sheath", "scabbard", "holster", "weapon", "pouch", "skirt", "gun", "dagger"]):
                        self.detached_hip_accessories.append(child_idx)

        self.node_translations: Dict[str, np.ndarray] = {}
        self._build_finger_curl_axes()

    def _build_topological_order(self) -> List[int]:
        """Compute topological ordering of glTF nodes so parents strictly precede children."""
        order: List[int] = []
        visited: Set[int] = set()

        def visit(i: int) -> None:
            p = self.parent_of.get(i)
            if p is not None and p not in visited:
                visit(p)
            if i not in visited:
                visited.add(i)
                order.append(i)

        for i in range(len(self.nodes)):
            visit(i)
        return order

    def _build_topological_aim_axes(self) -> None:
        """Discover each bone's physical heading vector in its local coordinate frame."""
        for canon_p, canon_c in LIMB_CHILD.items():
            canon_c_actual = canon_c
            if canon_p == "spine" and "spine1" not in self.rig_info.bone_mapping:
                canon_c_actual = "chest"
            if canon_p in self.rig_info.bone_mapping and canon_c_actual in self.rig_info.bone_mapping:
                p_b = self.rig_info.bone_mapping[canon_p]
                c_b = self.rig_info.bone_mapping[canon_c_actual]
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

        # For feet, discover along-local axis (ankle -> toe) from child node or rest direction
        for canon_foot in ["left_foot", "right_foot"]:
            if canon_foot in self.rig_info.bone_mapping:
                tb = self.rig_info.bone_mapping[canon_foot]
                if tb in self.name_to_idx:
                    idx = self.name_to_idx[tb]
                    ch_list = self.children_of.get(idx, [])
                    m_rot = _orthonormalize_matrix(self.rest_world_mat[idx][:3, :3])
                    if ch_list:
                        ch_idx = ch_list[0]
                        v_w = self.rest_world_mat[ch_idx][:3, 3] - self.rest_world_mat[idx][:3, 3]
                        if float(np.linalg.norm(v_w)) > 1e-5:
                            self.aim_local[canon_foot] = m_rot.T @ normalize(v_w)
                    if canon_foot not in self.aim_local:
                        self.aim_local[canon_foot] = m_rot.T @ np.array([0.0, 0.0, 1.0])

        # For toes, discover aim axis from toe child (tip) or forward default
        for canon_toe in ["left_toe", "right_toe"]:
            if canon_toe in self.rig_info.bone_mapping:
                tb = self.rig_info.bone_mapping[canon_toe]
                if tb in self.name_to_idx:
                    idx = self.name_to_idx[tb]
                    ch_list = self.children_of.get(idx, [])
                    m_rot = _orthonormalize_matrix(self.rest_world_mat[idx][:3, :3])
                    if ch_list:
                        ch_idx = ch_list[0]
                        v_w = self.rest_world_mat[ch_idx][:3, 3] - self.rest_world_mat[idx][:3, 3]
                        if float(np.linalg.norm(v_w)) > 1e-5:
                            self.aim_local[canon_toe] = m_rot.T @ normalize(v_w)
                    if canon_toe not in self.aim_local:
                        self.aim_local[canon_toe] = m_rot.T @ np.array([0.0, 0.0, 1.0])

        # For hands, discover along-local axis from child node or forearm direction
        for canon_hand in ["left_hand", "right_hand"]:
            if canon_hand in self.rig_info.bone_mapping:
                tb = self.rig_info.bone_mapping[canon_hand]
                if tb in self.name_to_idx:
                    idx = self.name_to_idx[tb]
                    ch_list = self.children_of.get(idx, [])
                    m_rot = _orthonormalize_matrix(self.rest_world_mat[idx][:3, :3])
                    if ch_list:
                        ch_idx = ch_list[0]
                        v_w = self.rest_world_mat[ch_idx][:3, 3] - self.rest_world_mat[idx][:3, 3]
                        if float(np.linalg.norm(v_w)) > 1e-5:
                            self.aim_local[canon_hand] = m_rot.T @ normalize(v_w)
                    if canon_hand not in self.aim_local:
                        lat = 1.0 if "left" in canon_hand else -1.0
                        self.aim_local[canon_hand] = m_rot.T @ np.array([lat, 0.0, 0.0])

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

    def _build_finger_curl_axes(self) -> None:
        """Precompute anatomically correct local flexion curl axes for every finger bone.
        
        Uses PCA knuckle span orientation across the palm for fingers, and opposition to
        palm center for thumbs, with sign calibrated to decrease distance to the wrist.
        """
        self.finger_curl_axes: Dict[int, np.ndarray] = {}
        for side in ["left", "right"]:
            w_bone = self.rig_info.bone_mapping.get(f"{side}_hand")
            w_idx = self.name_to_idx.get(w_bone)
            p_wrist = self.rest_world_mat[w_idx][:3, 3] if w_idx is not None else np.zeros(3)

            knuckle_indices = [f[0] for f in self.finger_bones if f[1] == side and f[2] == 1 and not f[3]]
            pts = [self.rest_world_mat[k][:3, 3] for k in knuckle_indices]
            p_palm_center = np.mean(pts, axis=0) if pts else p_wrist

            if len(knuckle_indices) >= 2:
                pts_centered = pts - np.mean(pts, axis=0)
                _, _, vt = np.linalg.svd(pts_centered)
                v_knuckle_span = vt[0]
            else:
                v_knuckle_span = np.array([0.0, 0.0, 1.0])

            for f_idx, s, phalanx, is_thumb in self.finger_bones:
                if s != side:
                    continue
                p_bone = self.rest_world_mat[f_idx][:3, 3]
                ch_list = self.children_of.get(f_idx, [])
                if ch_list and float(np.linalg.norm(self.rest_local_trans[ch_list[0]])) > 1e-4:
                    d_aim = normalize(self.rest_local_trans[ch_list[0]])
                    b_len = float(np.linalg.norm(self.rest_local_trans[ch_list[0]]))
                else:
                    d_aim = np.array([1.0, 0.0, 0.0]) if abs(self.rest_local_mat[f_idx][0, 3]) > abs(self.rest_local_mat[f_idx][1, 3]) else np.array([0.0, 1.0, 0.0])
                    b_len = 1.0
                w_rot = _orthonormalize_matrix(self.rest_world_mat[f_idx][:3, :3])

                if not is_thumb:
                    axis_k = w_rot.T @ v_knuckle_span
                    axis_curl = normalize(axis_k - float(np.dot(axis_k, d_aim)) * d_aim)
                    # Calibrate curl sign so finger curls towards wrist
                    rad = np.radians(35.0)
                    c, s_val = np.cos(rad), np.sin(rad)
                    u = axis_curl
                    r_pos = np.eye(3) * c + (1 - c) * np.outer(u, u) + np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_val
                    r_neg = np.eye(3) * c + (1 - c) * np.outer(u, u) - np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_val
                    tip_pos = p_bone + w_rot @ (r_pos @ d_aim * b_len)
                    tip_neg = p_bone + w_rot @ (r_neg @ d_aim * b_len)
                    if np.linalg.norm(tip_pos - p_wrist) > np.linalg.norm(tip_neg - p_wrist):
                        axis_curl = -axis_curl
                else:
                    v_to_palm = normalize(w_rot.T @ (p_palm_center - p_bone))
                    axis_curl = normalize(np.cross(d_aim, v_to_palm))
                    if float(np.linalg.norm(axis_curl)) < 1e-3:
                        axis_curl = np.array([0.0, 0.0, 1.0])
                    rad = np.radians(35.0)
                    c, s_val = np.cos(rad), np.sin(rad)
                    u = axis_curl
                    r_pos = np.eye(3) * c + (1 - c) * np.outer(u, u) + np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_val
                    r_neg = np.eye(3) * c + (1 - c) * np.outer(u, u) - np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_val
                    tip_pos = p_bone + w_rot @ (r_pos @ d_aim * b_len)
                    tip_neg = p_bone + w_rot @ (r_neg @ d_aim * b_len)
                    if np.linalg.norm(tip_pos - p_palm_center) > np.linalg.norm(tip_neg - p_palm_center):
                        axis_curl = -axis_curl

                self.finger_curl_axes[f_idx] = axis_curl

    def transfer_motion(
        self,
        canonical_motion: CanonicalMotion,
        raw_motion: Optional[Any] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Transfer 3D kinematics onto target rig bones using Motion2MixamoRig reference FK & alignment.

        Returns:
            bone_rotations: target_bone_name -> (F, 4) quaternions [x, y, z, w]
            root_translations: (F, 3) translations [x, y, z] for hips
        """
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
            joint_pos = {}

        # Root translation for hips: contact grounding + parent space projection
        root_bone = self.rig_info.bone_mapping.get("hips")
        root_trans = np.zeros((num_frames, 3), dtype=np.float32)
        m_p_inv = np.eye(4, dtype=np.float64)
        if root_bone and root_bone in self.name_to_idx:
            idx = self.name_to_idx[root_bone]
            p_idx = self.parent_of.get(idx)
            m_p_inv = np.linalg.inv(self.rest_world_mat[p_idx]) if p_idx is not None else np.eye(4, dtype=np.float64)
            scale_factor = float(self.model_height / 1.75) if self.model_height > 0 else 1.0

            if "left_foot" in joint_pos and "right_foot" in joint_pos and "hips" in joint_pos:
                y_stance_0 = min(float(joint_pos["left_foot"][0, 1]), float(joint_pos["right_foot"][0, 1]))
                for f in range(num_frames):
                    y_stance_f = min(float(joint_pos["left_foot"][f, 1]), float(joint_pos["right_foot"][f, 1]))
                    dy_ground = -(y_stance_f - y_stance_0)
                    dx = float(joint_pos["hips"][f, 0] - joint_pos["hips"][0, 0])
                    dz = float(joint_pos["hips"][f, 2] - joint_pos["hips"][0, 2])
                    v_w = np.array([dx * scale_factor, dy_ground * scale_factor, dz * scale_factor], dtype=np.float64)
                    v_local = m_p_inv[:3, :3] @ v_w
                    root_trans[f] = v_local.astype(np.float32)
            elif raw_motion is not None and hasattr(raw_motion, "root_translation"):
                raw_t = raw_motion.root_translation.astype(np.float64)
                for f in range(num_frames):
                    v_w = (raw_t[f] - raw_t[0]) * scale_factor
                    v_local = m_p_inv[:3, :3] @ v_w
                    root_trans[f] = v_local.astype(np.float32)

        self.node_translations = {}
        if root_bone:
            self.node_translations[root_bone] = root_trans

        # Prepare IK tracks for detached foot bones (e.g. FootIK in Rigify/Blender)
        ik_translations: Dict[int, np.ndarray] = {
            ik_idx: np.zeros((num_frames, 3), dtype=np.float32)
            for _, ik_idx, _ in self.detached_foot_nodes.values()
        }

        for canonical, target_bone in self.rig_info.bone_mapping.items():
            if target_bone in self.name_to_idx:
                bone_rotations[target_bone] = np.zeros((num_frames, 4), dtype=np.float32)
                idx = self.name_to_idx[target_bone]
                q_init = self.rest_local_quat[idx].astype(np.float32)
                prev_quat[target_bone] = q_init
                prev_rot_mat[target_bone] = quat_xyzw_to_matrix(q_init)

        # Initialize rotation tracks for procedural helper twist and share bones
        for h_idx in list(self.twist_helpers.keys()) + list(self.share_bones.keys()):
            h_name = self.idx_to_name[h_idx]
            bone_rotations[h_name] = np.zeros((num_frames, 4), dtype=np.float32)
            q_init = self.rest_local_quat[h_idx].astype(np.float32)
            prev_quat[h_name] = q_init
            prev_rot_mat[h_name] = quat_xyzw_to_matrix(q_init)

        # Initialize rotation tracks for procedural finger bones
        for f_idx, _, _, _ in self.finger_bones:
            f_name = self.idx_to_name[f_idx]
            bone_rotations[f_name] = np.zeros((num_frames, 4), dtype=np.float32)
            q_init = self.rest_local_quat[f_idx].astype(np.float32)
            prev_quat[f_name] = q_init
            prev_rot_mat[f_name] = quat_xyzw_to_matrix(q_init)

        # Fully determined bones where twist continuity must not override direct 3D orientation
        fully_determined: Set[str] = {"hips", "chest", "left_foot", "right_foot", "left_toe", "right_toe", "spine1"}

        # Precompute per-side dynamic finger curl trajectories with temporal filtering
        curl_dyn_by_side: Dict[str, np.ndarray] = {
            "left": np.zeros(num_frames, dtype=np.float64),
            "right": np.zeros(num_frames, dtype=np.float64),
        }
        for side in ["left", "right"]:
            w_key = f"{side}_hand"
            fa_key = f"{side}_forearm"
            i_key = f"{side}_fingers"
            p_key = f"{side}_pinky"

            if w_key in joint_pos and i_key in joint_pos:
                raw_curls = []
                for f in range(num_frames):
                    d_fa = float(np.linalg.norm(joint_pos[w_key][f] - joint_pos[fa_key][f])) if fa_key in joint_pos else 0.22
                    d_fa = max(d_fa, 0.10)
                    r_idx = float(np.linalg.norm(joint_pos[i_key][f] - joint_pos[w_key][f])) / d_fa
                    r_pky = float(np.linalg.norm(joint_pos[p_key][f] - joint_pos[w_key][f])) / d_fa if p_key in joint_pos else r_idx
                    # Maximum hand extension span across index and pinky (scaled to match index reach)
                    span = max(r_idx, r_pky * 1.35)
                    # Natural calibrated extension thresholds:
                    # Extended/open hand is span >= 0.52 (curl = 0.0)
                    # Tightly clenched fist is span <= 0.33 (curl = 1.0)
                    c = float(np.clip((0.52 - span) / (0.52 - 0.33), 0.0, 1.0))
                    raw_curls.append(c)

                # Smooth with EMA filter (alpha=0.35) to eliminate single-frame tracking noise
                s = raw_curls[0] if raw_curls else 0.0
                for f, c in enumerate(raw_curls):
                    s = 0.35 * c + 0.65 * s
                    curl_dyn_by_side[side][f] = s

        # Solve per-frame hierarchical rotations with Forward Kinematics propagation
        for f in range(num_frames):
            # Local rotations and translations for all nodes this frame (initialized to rest pose)
            curr_local_rot: Dict[int, np.ndarray] = {
                i: quat_xyzw_to_matrix(self.rest_local_quat[i]) for i in range(len(self.nodes))
            }
            curr_local_trans: Dict[int, np.ndarray] = {
                i: self.rest_local_trans[i].copy() for i in range(len(self.nodes))
            }
            if root_bone and root_bone in self.name_to_idx:
                hips_idx = self.name_to_idx[root_bone]
                curr_local_trans[hips_idx] += root_trans[f].astype(np.float64)

            curr_world_rot: Dict[int, np.ndarray] = {}
            curr_world_mat: Dict[int, np.ndarray] = {}

            def _update_fk() -> None:
                """Propagate world transforms down the topological hierarchy."""
                for node_i in self.topological_order:
                    p_i = self.parent_of.get(node_i)
                    loc_r = curr_local_rot[node_i]
                    loc_t = curr_local_trans[node_i]
                    loc_m = np.eye(4, dtype=np.float64)
                    loc_m[:3, :3] = loc_r
                    loc_m[:3, 3] = loc_t
                    if p_i is None:
                        curr_world_rot[node_i] = _orthonormalize_matrix(loc_r)
                        curr_world_mat[node_i] = loc_m
                    else:
                        w_m = curr_world_mat[p_i] @ loc_m
                        curr_world_mat[node_i] = w_m
                        curr_world_rot[node_i] = _orthonormalize_matrix(w_m[:3, :3])

            _update_fk()

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
                        r_tgt_frame = orthonormal_frame(v_lat, v_up)
                        # Rotate model's rest orientation into the current human target frame
                        r_hips_world = (r_tgt_frame @ self.r_pelvis_rest.T) @ w_bone_rest
                        r_local = w_parent.T @ r_hips_world
                    elif f < len(canonical_motion.r_anim):
                        r_canon_f = canonical_motion.r_anim[f].get("hips", np.eye(3))
                        r_local = r_bone_rest @ r_canon_f

                elif canon == "spine1":
                    if "spine1" in joint_pos and "chest" in joint_pos:
                        v_human = joint_pos["chest"][f] - joint_pos["spine1"][f]
                    elif "spine" in joint_pos and "chest" in joint_pos:
                        v_human = joint_pos["chest"][f] - joint_pos["spine"][f]
                    else:
                        v_human = np.array([0.0, 1.0, 0.0])
                    d_aim = self.aim_local.get("spine1", np.array([0.0, 1.0, 0.0]))
                    target_w = normalize(v_human, np.array([0.0, 1.0, 0.0]))
                    target_in_parent = w_parent.T @ target_w
                    r_delta = rotation_aligning(r_bone_rest @ d_aim, target_in_parent)
                    r_local = r_delta @ r_bone_rest

                elif canon == "chest":
                    if "left_shoulder" in joint_pos and "right_shoulder" in joint_pos and "neck" in joint_pos and "spine" in joint_pos:
                        v_lat = joint_pos["left_shoulder"][f] - joint_pos["right_shoulder"][f]
                        v_up = joint_pos["neck"][f] - joint_pos["spine"][f]
                        r_tgt_chest_frame = orthonormal_frame(v_lat, v_up)
                        r_chest_world = (r_tgt_chest_frame @ self.r_chest_rest.T) @ w_bone_rest
                        r_local = w_parent.T @ r_chest_world
                    elif f < len(canonical_motion.r_anim):
                        r_canon_f = canonical_motion.r_anim[f].get("chest", np.eye(3))
                        r_local = r_bone_rest @ r_canon_f

                elif canon in ["left_foot", "right_foot"]:
                    # Ported _foot_rotation from Motion2MixamoRig
                    side_leg = "left_leg" if "left" in canon else "right_leg"
                    toe_key = "left_toe" if "left" in canon else "right_toe"
                    along_local = self.aim_local.get(canon, np.array([0.0, 0.0, 1.0]))
                    r_rest_frame = frame_from_primary_secondary(w_bone_rest @ along_local, np.array([0.0, 1.0, 0.0]))

                    if toe_key in joint_pos and canon in joint_pos:
                        v_foot = joint_pos[toe_key][f] - joint_pos[canon][f]
                    elif canon in joint_pos and side_leg in joint_pos:
                        v_foot = joint_pos[canon][f] - joint_pos[side_leg][f]
                    else:
                        v_foot = np.array([0.0, 0.0, 1.0])

                    tgt_along = normalize(v_foot, np.array([0.0, 0.0, 1.0]))
                    tgt_up = np.array([0.0, 1.0, 0.0])
                    if abs(float(np.dot(tgt_along, tgt_up))) > 0.92:
                        side_upleg = "left_upleg" if "left" in canon else "right_upleg"
                        if side_upleg in joint_pos and side_leg in joint_pos:
                            tgt_up = normalize(joint_pos[side_leg][f] - joint_pos[side_upleg][f], np.array([0.0, 1.0, 0.0]))
                    r_tgt_frame = frame_from_primary_secondary(tgt_along, tgt_up)
                    r_foot_world = (r_tgt_frame @ r_rest_frame.T) @ w_bone_rest
                    r_local = w_parent.T @ r_foot_world

                elif canon in ["left_toe", "right_toe"]:
                    side_foot = "left_foot" if "left" in canon else "right_foot"
                    if canon in joint_pos and side_foot in joint_pos:
                        dy_toe = float(joint_pos[side_foot][f, 1] - joint_pos[canon][f, 1])
                    else:
                        dy_toe = 0.0
                    pitch_rad = float(np.arcsin(np.clip(dy_toe / 0.12, 0.0, 0.70))) if dy_toe > 0.02 else 0.0
                    along_local = self.aim_local.get(canon, np.array([0.0, 0.0, 1.0]))
                    axis_lat = normalize(np.cross(np.array([0.0, 1.0, 0.0]), along_local), np.array([1.0, 0.0, 0.0]))
                    if abs(pitch_rad) > 1e-4:
                        c_p, s_p = np.cos(pitch_rad), np.sin(pitch_rad)
                        u = axis_lat
                        r_pitch = np.eye(3) * c_p + (1.0 - c_p) * np.outer(u, u) + np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_p
                        r_local = r_bone_rest @ r_pitch
                    else:
                        r_local = r_bone_rest

                elif canon in ["left_hand", "right_hand"]:
                    # Aim hand along wrist -> fingers direction
                    side_fa = "left_forearm" if "left" in canon else "right_forearm"
                    fingers_key = "left_fingers" if "left" in canon else "right_fingers"
                    if fingers_key in joint_pos and canon in joint_pos:
                        v_hand = joint_pos[fingers_key][f] - joint_pos[canon][f]
                    elif canon in joint_pos and side_fa in joint_pos:
                        v_hand = joint_pos[canon][f] - joint_pos[side_fa][f]
                    else:
                        v_hand = np.zeros(3)

                    if float(np.linalg.norm(v_hand)) > 1e-5:
                        lat = 1.0 if "left" in canon else -1.0
                        target_w = normalize(v_hand, np.array([lat, 0.0, 0.0]))
                        d_aim = self.aim_local.get(canon, np.array([lat, 0.0, 0.0]))
                        target_in_parent = w_parent.T @ target_w
                        r_delta = rotation_aligning(r_bone_rest @ d_aim, target_in_parent)
                        r_local = r_delta @ r_bone_rest
                    else:
                        r_local = r_bone_rest

                elif canon in self.aim_local and canon in LIMB_HUMAN_PAIRS:
                    # Swing limbs: arms, legs, spine, neck, head
                    d_aim = self.aim_local[canon]
                    hp_p, hp_c = LIMB_HUMAN_PAIRS[canon]
                    if hp_p in joint_pos and hp_c in joint_pos:
                        pos_p = joint_pos[hp_p][f]
                        pos_c = joint_pos[hp_c][f]
                        v_human = pos_c - pos_p
                        if float(np.linalg.norm(v_human)) > 1e-5:
                            target_w = normalize(v_human, np.array([0.0, 1.0, 0.0]))
                            target_in_parent = w_parent.T @ target_w
                            r_delta = rotation_aligning(r_bone_rest @ d_aim, target_in_parent)
                            r_local = r_delta @ r_bone_rest
                        else:
                            r_local = r_bone_rest
                    else:
                        r_local = r_bone_rest

                elif f < len(canonical_motion.r_anim):
                    d = canonical_motion.r_anim[f].get(canon, np.eye(3, dtype=np.float64))
                    c_mat = self.basis_change.get(target_bone, np.eye(3, dtype=np.float64))
                    d_prime = c_mat @ d @ c_mat.T
                    r_local = r_bone_rest @ d_prime

                # Geodesic anti-twist continuity (exempt fully determined bones; frame 0 never flips)
                if canon not in fully_determined and f > 0:
                    aim_axis = self.aim_local.get(canon, np.array([0.0, 1.0, 0.0], dtype=np.float64))
                    r_clean, _ = pick_closest_aim_twist(r_local, prev_rot_mat.get(target_bone), aim_axis)
                else:
                    r_clean = r_local

                prev_rot_mat[target_bone] = r_clean

                # Commit local rotation and propagate forward kinematics immediately
                curr_local_rot[idx] = r_clean
                _update_fk()

                # Quaternion sign continuity
                q_anim = matrix_to_quat_xyzw(r_clean)
                q_clean = continue_quaternion_xyzw(q_anim, prev_quat.get(target_bone))
                prev_quat[target_bone] = q_clean
                bone_rotations[target_bone][f] = q_clean

            # Procedural helper twist and share bone propagation
            for h_idx, (p_idx, weight, p_canon) in self.twist_helpers.items():
                r_prim_loc = curr_local_rot[p_idx]
                r_prim_rest = quat_xyzw_to_matrix(self.rest_local_quat[p_idx])
                r_delta = r_prim_loc @ r_prim_rest.T
                q_delta = matrix_to_quat_xyzw(r_delta)

                d_aim = self.aim_local.get(p_canon, np.array([0.0, 1.0, 0.0]))
                proj = float(np.dot(q_delta[:3], d_aim))
                q_twist = np.array([proj * d_aim[0], proj * d_aim[1], proj * d_aim[2], q_delta[3]], dtype=np.float64)
                norm_q = float(np.linalg.norm(q_twist))
                if norm_q > 1e-6:
                    q_twist /= norm_q
                    half_angle = float(np.arccos(np.clip(q_twist[3], -1.0, 1.0)))
                    frac_half = half_angle * weight
                    sin_h = float(np.sin(half_angle))
                    u_axis = q_twist[:3] / sin_h if abs(sin_h) > 1e-6 else d_aim
                    q_frac = np.array([u_axis[0] * np.sin(frac_half), u_axis[1] * np.sin(frac_half), u_axis[2] * np.sin(frac_half), np.cos(frac_half)], dtype=np.float64)
                    r_frac = quat_xyzw_to_matrix(q_frac)
                else:
                    r_frac = np.eye(3, dtype=np.float64)

                r_h_rest = quat_xyzw_to_matrix(self.rest_local_quat[h_idx])
                r_h_loc = r_h_rest @ r_frac
                curr_local_rot[h_idx] = r_h_loc
                h_name = self.idx_to_name[h_idx]
                q_clean = continue_quaternion_xyzw(matrix_to_quat_xyzw(r_h_loc), prev_quat.get(h_name))
                prev_quat[h_name] = q_clean
                bone_rotations[h_name][f] = q_clean

            for s_idx, (p_idx, weight, p_canon) in self.share_bones.items():
                r_prim_loc = curr_local_rot[p_idx]
                r_prim_rest = quat_xyzw_to_matrix(self.rest_local_quat[p_idx])
                r_delta = r_prim_loc @ r_prim_rest.T
                q_delta = matrix_to_quat_xyzw(r_delta)
                half_angle = float(np.arccos(np.clip(q_delta[3], -1.0, 1.0)))
                frac_half = half_angle * weight
                sin_h = float(np.sin(half_angle))
                u_axis = q_delta[:3] / sin_h if abs(sin_h) > 1e-6 else np.array([0.0, 1.0, 0.0])
                q_frac = np.array([u_axis[0] * np.sin(frac_half), u_axis[1] * np.sin(frac_half), u_axis[2] * np.sin(frac_half), np.cos(frac_half)], dtype=np.float64)
                r_s_rest = quat_xyzw_to_matrix(self.rest_local_quat[s_idx])
                r_s_loc = r_s_rest @ quat_xyzw_to_matrix(q_frac)
                curr_local_rot[s_idx] = r_s_loc
                s_name = self.idx_to_name[s_idx]
                q_clean = continue_quaternion_xyzw(matrix_to_quat_xyzw(r_s_loc), prev_quat.get(s_name))
                prev_quat[s_name] = q_clean
                bone_rotations[s_name][f] = q_clean

            # Procedural finger posing and dynamic closed fist flexion
            for f_idx, side, phalanx, is_thumb in self.finger_bones:
                curl_dyn = float(curl_dyn_by_side.get(side, np.zeros(num_frames))[f])

                if is_thumb:
                    deg = 4.0 + 54.0 * curl_dyn
                else:
                    if phalanx == 1:
                        deg = 4.0 + 60.0 * curl_dyn
                    elif phalanx == 2:
                        deg = 6.0 + 74.0 * curl_dyn
                    else:
                        deg = 4.0 + 52.0 * curl_dyn
                rad = float(np.radians(deg))

                axis_curl = self.finger_curl_axes.get(f_idx, np.array([0.0, 0.0, 1.0]))
                c_a, s_a = np.cos(rad), np.sin(rad)
                u = axis_curl
                r_curl = np.eye(3) * c_a + (1.0 - c_a) * np.outer(u, u) + np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) * s_a
                r_f_rest = quat_xyzw_to_matrix(self.rest_local_quat[f_idx])
                r_f_loc = r_f_rest @ r_curl

                curr_local_rot[f_idx] = r_f_loc
                f_name = self.idx_to_name[f_idx]
                q_clean = continue_quaternion_xyzw(matrix_to_quat_xyzw(r_f_loc), prev_quat.get(f_name))
                prev_quat[f_name] = q_clean
                bone_rotations[f_name][f] = q_clean

            # Update FK with procedural bones
            _update_fk()

            # Animate detached IK foot nodes so meshes stay 100% attached at the ankles
            for canon_foot, (f_idx, ik_idx, l_idx) in self.detached_foot_nodes.items():
                m_knee_rest = self.rest_world_mat[l_idx]
                p_foot_rest = self.rest_world_pos.get(canon_foot, self.rest_world_mat[f_idx][:3, 3])
                v_shin_local = (np.linalg.inv(m_knee_rest) @ np.append(p_foot_rest, 1.0))[:3]
                p_ankle_world = (curr_world_mat[l_idx] @ np.append(v_shin_local, 1.0))[:3]

                ik_p = self.parent_of.get(ik_idx)
                m_ik_p = curr_world_mat[ik_p] if ik_p is not None else np.eye(4, dtype=np.float64)
                t_ik_local = (np.linalg.inv(m_ik_p) @ np.append(p_ankle_world, 1.0))[:3]
                delta_ik = t_ik_local - self.rest_local_trans[ik_idx]
                ik_translations[ik_idx][f] = delta_ik.astype(np.float32)

        for ik_idx, track in ik_translations.items():
            ik_name = self.idx_to_name[ik_idx]
            self.node_translations[ik_name] = track

        # Animate detached hip accessories so they follow the hips across the floor
        for acc_idx in self.detached_hip_accessories:
            acc_name = self.idx_to_name[acc_idx]
            self.node_translations[acc_name] = root_trans

        # Retain static baseline folded rotations for any accessory bones
        for acc_idx, acc_q in self.accessory_baseline_rot.items():
            acc_name = self.idx_to_name[acc_idx]
            if acc_name not in bone_rotations:
                bone_rotations[acc_name] = np.tile(acc_q.astype(np.float32), (num_frames, 1))

        return bone_rotations, root_trans
