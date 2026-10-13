"""Universal 3D Model Inspector and Rig Detector.

Analyzes GLB, GLTF, FBX, and OBJ files to determine if a model has an existing
armature (rigged) or is a static mesh (unrigged).
Maps detected bones into a standardized canonical humanoid hierarchy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class RigType(str, Enum):
    MIXAMO = "mixamo"
    VRM_HUMANOID = "vrm_humanoid"
    BLENDER_RIGIFY = "blender_rigify"
    CHARACTER_CREATOR = "character_creator"
    UNREAL = "unreal_mannequin"
    GENERIC = "generic_humanoid"
    UNRIGGED = "unrigged_mesh"


# Canonical semantic humanoid joint names
CANONICAL_JOINTS = (
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
)

# Known aliases across 3D ecosystems (Mixamo, VRM, Rigify, Unreal, Reallusion Character Creator, Daz3D)
JOINT_ALIASES: Dict[str, List[str]] = {
    "hips": ["mixamorig:hips", "mixamorig_hips", "hips", "pelvis", "root", "hip", "b_hips", "cc_base_pelvis", "cc_base_hip"],
    "spine": ["mixamorig:spine", "mixamorig_spine", "spine", "spine01", "spine_01", "spine1", "def-spine", "b_spine", "cc_base_spine01"],
    "chest": ["mixamorig:spine2", "mixamorig:spine1", "chest", "upperchest", "spine02", "spine_02", "spine_03", "spine2", "def-chest", "b_chest", "cc_base_spine02"],
    "neck": ["mixamorig:neck", "mixamorig_neck", "neck", "neck_01", "neck01", "def-neck", "b_neck", "cc_base_neck", "cc_base_necktwist01"],
    "head": ["mixamorig:head", "mixamorig_head", "head", "head_01", "def-head", "b_head", "cc_base_head"],
    "left_shoulder": ["mixamorig:leftshoulder", "leftshoulder", "clavicle_l", "l_clavicle", "shoulder.l", "def-shoulder.l", "b_l_shoulder", "cc_base_l_clavicle"],
    "left_arm": ["mixamorig:leftarm", "leftupperarm", "upperarm_l", "upper_arm.l", "arm.l", "def-upper_arm.l", "b_l_arm", "l_upperarm", "l_arm", "cc_base_l_upperarm"],
    "left_forearm": ["mixamorig:leftforearm", "leftlowerarm", "lowerarm_l", "forearm.l", "def-forearm.l", "b_l_forearm", "l_forearm", "l_lowerarm", "cc_base_l_forearm"],
    "left_hand": ["mixamorig:lefthand", "lefthand", "hand_l", "hand.l", "def-hand.l", "b_l_hand", "l_hand", "cc_base_l_hand"],
    "right_shoulder": ["mixamorig:rightshoulder", "rightshoulder", "clavicle_r", "r_clavicle", "shoulder.r", "def-shoulder.r", "b_r_shoulder", "cc_base_r_clavicle"],
    "right_arm": ["mixamorig:rightarm", "rightupperarm", "upperarm_r", "upper_arm.r", "arm.r", "def-upper_arm.r", "b_r_arm", "r_upperarm", "r_arm", "cc_base_r_upperarm"],
    "right_forearm": ["mixamorig:rightforearm", "rightlowerarm", "lowerarm_r", "forearm.r", "def-forearm.r", "b_r_forearm", "r_forearm", "r_lowerarm", "cc_base_r_forearm"],
    "right_hand": ["mixamorig:righthand", "righthand", "hand_r", "hand.r", "def-hand.r", "b_r_hand", "r_hand", "cc_base_r_hand"],
    "left_upleg": ["mixamorig:leftupleg", "leftupperleg", "thigh_l", "thigh.l", "def-thigh.l", "b_l_upleg", "l_thigh", "l_upperleg", "cc_base_l_thigh"],
    "left_leg": ["mixamorig:leftleg", "leftlowerleg", "calf_l", "shin.l", "def-shin.l", "b_l_leg", "l_calf", "l_lowerleg", "l_leg", "cc_base_l_calf"],
    "left_foot": ["mixamorig:leftfoot", "leftfoot", "foot_l", "foot.l", "def-foot.l", "b_l_foot", "l_foot", "cc_base_l_foot"],
    "right_upleg": ["mixamorig:rightupleg", "rightupperleg", "thigh_r", "thigh.r", "def-thigh.r", "b_r_upleg", "r_thigh", "r_upperleg", "cc_base_r_thigh"],
    "right_leg": ["mixamorig:rightleg", "rightlowerleg", "calf_r", "shin.r", "def-shin.r", "b_r_leg", "r_calf", "r_lowerleg", "r_leg", "cc_base_r_calf"],
    "right_foot": ["mixamorig:rightfoot", "rightfoot", "foot_r", "foot.r", "def-foot.r", "b_r_foot", "r_foot", "cc_base_r_foot"],
}


@dataclass
class RigAnalysis:
    """Detailed summary of the analyzed 3D character asset."""
    file_path: Path
    format: str
    is_rigged: bool
    rig_type: RigType
    all_bones: List[str] = field(default_factory=list)
    bone_mapping: Dict[str, str] = field(default_factory=dict)  # canonical -> target bone name
    vertex_count: int = 0
    warnings: List[str] = field(default_factory=list)
    bind_rotations: Dict[str, List[float]] = field(default_factory=dict)
    rest_arm_angle_deg: float = 0.0
    detected_rest_pose: str = "t_pose"


def parse_glb_json(glb_path: Path) -> tuple[dict[str, Any], int]:
    """Parse the JSON chunk from a binary GLB file without external heavy dependencies."""
    with open(glb_path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"glTF":
            raise ValueError(f"File {glb_path.name} is not a valid GLTF 2.0 binary (.glb)")
        
        chunk_len = int.from_bytes(f.read(4), "little")
        chunk_type = f.read(4)
        if chunk_type != b"JSON":
            raise ValueError("Expected JSON chunk in GLB")
        
        json_data = f.read(chunk_len).decode("utf-8")
        gltf = json.loads(json_data)
        
        # Estimate vertex count from accessors if available
        total_verts = 0
        for acc in gltf.get("accessors", []):
            if acc.get("type") == "VEC3":
                total_verts = max(total_verts, acc.get("count", 0))
                
        return gltf, total_verts


def inspect_model(file_path: str | Path) -> RigAnalysis:
    """Analyze a 3D model asset (GLB, GLTF, FBX, OBJ)."""
    p = Path(file_path)
    suffix = p.suffix.lower()

    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    if suffix in [".glb", ".gltf"]:
        return _inspect_gltf_or_glb(p)
    elif suffix == ".obj":
        # Raw OBJ files have no armature or skinning weights
        return RigAnalysis(
            file_path=p,
            format="OBJ",
            is_rigged=False,
            rig_type=RigType.UNRIGGED,
            warnings=["OBJ is a static format without an armature. Experimental Auto-Rigging required."],
        )
    elif suffix == ".fbx":
        return _inspect_fbx(p)
    else:
        return RigAnalysis(
            file_path=p,
            format=suffix.upper(),
            is_rigged=False,
            rig_type=RigType.UNRIGGED,
            warnings=[f"Unknown or unsupported extension: {suffix}."],
        )


def _inspect_gltf_or_glb(path: Path) -> RigAnalysis:
    try:
        if path.suffix.lower() == ".glb":
            gltf, vertex_count = parse_glb_json(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                gltf = json.load(f)
            vertex_count = 0
    except Exception as e:
        return RigAnalysis(
            file_path=path,
            format=path.suffix.upper(),
            is_rigged=False,
            rig_type=RigType.UNRIGGED,
            warnings=[f"Failed to parse GLTF/GLB: {e}"],
        )

    nodes = gltf.get("nodes", [])
    skins = gltf.get("skins", [])

    if not skins:
        return RigAnalysis(
            file_path=path,
            format="GLTF/GLB",
            is_rigged=False,
            rig_type=RigType.UNRIGGED,
            vertex_count=vertex_count,
            warnings=["No 'skins' definition found in GLTF. This is a static unrigged mesh."],
        )

    # Collect joint names and initial bind rotations
    skin = skins[0]
    joint_indices = skin.get("joints", [])
    bone_names: List[str] = []
    bind_rotations: Dict[str, List[float]] = {}

    for idx in joint_indices:
        if 0 <= idx < len(nodes):
            node = nodes[idx]
            name = node.get("name", f"bone_{idx}")
            bone_names.append(name)
            if "rotation" in node:
                bind_rotations[name] = node["rotation"]

    # Classify rig type and build mapping via Universal Semantic Matcher
    rig_type, mapping = _classify_bones(bone_names)

    # Compute 3D forward kinematics and topological solving
    world_pos, _ = TopologicalRigSolver.compute_world_transforms(nodes)

    # If missing core joints or low mapping count, invoke topological & geometric solver
    core_limbs = ["left_arm", "right_arm", "left_leg", "right_leg", "hips", "head"]
    if len(mapping) < 14 or any(j not in mapping for j in core_limbs):
        mapping = TopologicalRigSolver.solve_missing_joints(
            nodes=nodes,
            joint_indices=joint_indices,
            existing_mapping=mapping,
            world_pos=world_pos,
        )

    # Detect rest arm angle and classify rest pose
    rest_arm_angle_deg = TopologicalRigSolver.calculate_rest_arm_angle(
        nodes=nodes,
        mapping=mapping,
        world_pos=world_pos,
    )
    detected_rest_pose = "t_pose" if rest_arm_angle_deg < 12.0 else "a_pose"

    warnings = []
    missing_core = [j for j in ["hips", "spine", "head", "left_arm", "right_arm"] if j not in mapping]
    if missing_core:
        warnings.append(f"Missing core humanoid bones in detected rig: {missing_core}")

    return RigAnalysis(
        file_path=path,
        format="GLTF/GLB",
        is_rigged=True,
        rig_type=rig_type,
        all_bones=bone_names,
        bone_mapping=mapping,
        vertex_count=vertex_count,
        warnings=warnings,
        bind_rotations=bind_rotations,
        rest_arm_angle_deg=rest_arm_angle_deg,
        detected_rest_pose=detected_rest_pose,
    )


def _inspect_fbx(path: Path) -> RigAnalysis:
    """Lightweight FBX header/text scanner to identify bones when full FBX SDK is not present."""
    bone_names: List[str] = []
    try:
        content = path.read_bytes()
        # Scan for Mixamo or standard Model limb markers
        text_sample = content[: min(len(content), 1000000)].decode("latin-1", errors="ignore")
        for alias_list in JOINT_ALIASES.values():
            for alias in alias_list:
                if alias.lower() in text_sample.lower():
                    bone_names.append(alias)
        bone_names = list(set(bone_names))
    except Exception:
        pass

    if len(bone_names) >= 6:
        rig_type, mapping = _classify_bones(bone_names)
        return RigAnalysis(
            file_path=path,
            format="FBX",
            is_rigged=True,
            rig_type=rig_type,
            all_bones=bone_names,
            bone_mapping=mapping,
        )

    return RigAnalysis(
        file_path=path,
        format="FBX",
        is_rigged=False,
        rig_type=RigType.UNRIGGED,
        warnings=["No discernible humanoid armature found in FBX. Will treat as unrigged."],
    )


class UniversalBoneMatcher:
    """Intelligent NLP & Token-Based Anatomical Humanoid Bone Matcher.
    
    Operates without hardcoded rig assumptions: tokenizes any bone naming style
    (CamelCase, snake_case, dot.separated, prefixes, suffixes, Japanese Kanji/Romaji,
    and numbered limbs), extracts anatomical laterality (Left/Right/Center),
    and assigns optimal canonical matches while filtering auxiliary twist/share bones.
    """

    CANONICAL_ORDER: List[str] = [
        "left_shoulder", "right_shoulder",
        "left_hand", "right_hand",
        "left_forearm", "right_forearm",
        "left_arm", "right_arm",
        "left_foot", "right_foot",
        "left_upleg", "right_upleg",
        "left_leg", "right_leg",
        "head", "neck", "chest", "spine", "hips",
    ]

    CANONICAL_DEFS: Dict[str, Dict[str, Any]] = {
        "hips": {
            "side": "center",
            "high_keys": {"pelvis", "hip", "hips", "下半身", "センター", "腰"},
            "keys": {"waist", "root", "c_root"},
            "low_keys": {"rootjoint"},
            "bad": {"thigh", "leg", "toe", "foot", "arm", "spine", "head"},
        },
        "spine": {
            "side": "center",
            "high_keys": {"spine1", "spine01", "spine", "torso", "上半身"},
            "keys": {"abdomen", "lumbar"},
            "bad": {"chest", "spine2", "spine02", "spine3", "spine03", "neck", "head", "arm", "leg", "上半身2"},
        },
        "chest": {
            "side": "center",
            "high_keys": {"chest", "spine2", "spine02", "spine3", "spine03", "upperchest", "thorax", "上半身2"},
            "keys": {"pectoral", "upperbody"},
            "combo": [("spine", "02"), ("spine", "2"), ("spine", "03"), ("spine", "3")],
            "bad": {"neck", "head", "hip", "pelvis"},
        },
        "neck": {
            "side": "center",
            "high_keys": {"neck", "首"},
            "keys": {"throat", "cervical"},
            "bad": {"head", "chest", "spine"},
        },
        "head": {
            "side": "center",
            "high_keys": {"head", "頭"},
            "keys": {"skull", "cranium"},
            "bad": {"neck", "eye", "jaw", "teeth", "tongue", "ear", "hair"},
        },
        "left_shoulder": {
            "side": "left",
            "high_keys": {"clavicle", "shoulder", "collar", "shldr", "肩"},
            "keys": set(),
            "bad": {"arm", "forearm", "hand"},
        },
        "left_arm": {
            "side": "left",
            "high_keys": {"upperarm", "腕"},
            "keys": {"arm", "bicep", "brachium", "shldr"},
            "combo": [("upper", "arm"), ("up", "arm")],
            "bad": {"forearm", "lowerarm", "hand", "mid", "twist", "ひじ", "fore", "low", "lower", "elbow"},
        },
        "left_forearm": {
            "side": "left",
            "high_keys": {"forearm", "lowerarm", "ひじ"},
            "keys": {"elbow", "radius", "ulna"},
            "combo": [("lower", "arm"), ("fore", "arm"), ("low", "arm")],
            "bad": {"upper", "upperarm", "hand", "twist", "up"},
        },
        "left_hand": {
            "side": "left",
            "high_keys": {"hand", "手首"},
            "keys": {"wrist", "palm"},
            "bad": {"thumb", "index", "mid", "middle", "ring", "pinky", "finger", "toe", "arm", "forearm"},
        },
        "right_shoulder": {
            "side": "right",
            "high_keys": {"clavicle", "shoulder", "collar", "shldr", "肩"},
            "keys": set(),
            "bad": {"arm", "forearm", "hand"},
        },
        "right_arm": {
            "side": "right",
            "high_keys": {"upperarm", "腕"},
            "keys": {"arm", "bicep", "brachium", "shldr"},
            "combo": [("upper", "arm"), ("up", "arm")],
            "bad": {"forearm", "lowerarm", "hand", "mid", "twist", "ひじ", "fore", "low", "lower", "elbow"},
        },
        "right_forearm": {
            "side": "right",
            "high_keys": {"forearm", "lowerarm", "ひじ"},
            "keys": {"elbow", "radius", "ulna"},
            "combo": [("lower", "arm"), ("fore", "arm"), ("low", "arm")],
            "bad": {"upper", "upperarm", "hand", "twist", "up"},
        },
        "right_hand": {
            "side": "right",
            "high_keys": {"hand", "手首"},
            "keys": {"wrist", "palm"},
            "bad": {"thumb", "index", "mid", "middle", "ring", "pinky", "finger", "toe", "arm", "forearm"},
        },
        "left_upleg": {
            "side": "left",
            "high_keys": {"thigh", "upperleg", "upleg"},
            "keys": {"femur"},
            "combo": [("upper", "leg"), ("up", "leg")],
            "bad": {"calf", "shin", "foot", "twist", "ひざ", "lowerleg", "lower", "low"},
        },
        "left_leg": {
            "side": "left",
            "high_keys": {"calf", "shin", "lowerleg", "ひざ"},
            "keys": {"knee", "tibia", "leg", "足"},
            "combo": [("lower", "leg"), ("low", "leg")],
            "bad": {"thigh", "upperleg", "upleg", "up", "upper", "foot", "toe", "twist"},
        },
        "left_foot": {
            "side": "left",
            "high_keys": {"foot", "足首"},
            "keys": {"ankle", "heel"},
            "bad": {"toe", "thigh", "calf", "shin", "leg"},
        },
        "right_upleg": {
            "side": "right",
            "high_keys": {"thigh", "upperleg", "upleg"},
            "keys": {"femur"},
            "combo": [("upper", "leg"), ("up", "leg")],
            "bad": {"calf", "shin", "foot", "twist", "ひざ", "lowerleg", "lower", "low"},
        },
        "right_leg": {
            "side": "right",
            "high_keys": {"calf", "shin", "lowerleg", "ひざ"},
            "keys": {"knee", "tibia", "leg", "足"},
            "combo": [("lower", "leg"), ("low", "leg")],
            "bad": {"thigh", "upperleg", "upleg", "up", "upper", "foot", "toe", "twist"},
        },
        "right_foot": {
            "side": "right",
            "high_keys": {"foot", "足首"},
            "keys": {"ankle", "heel"},
            "bad": {"toe", "thigh", "calf", "shin", "leg"},
        },
    }

    @classmethod
    def tokenize(cls, name: str) -> List[str]:
        """Split name into alphanumeric sub-tokens across CamelCase and delimiters, including bigram compounds."""
        import re
        parts = re.split(r'[_.\-:\s|\\/]+', name)
        tokens: List[str] = []
        for p in parts:
            sub = re.findall(r'[A-Z]{2,}(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]|\d+', p)
            if sub:
                tokens.extend([s.lower() for s in sub])
            elif p:
                tokens.append(p.lower())
        
        compounds = set(tokens)
        for i in range(len(tokens) - 1):
            compounds.add(tokens[i] + tokens[i + 1])
        for i in range(len(tokens) - 2):
            compounds.add(tokens[i] + tokens[i + 1] + tokens[i + 2])
        return list(compounds)

    @staticmethod
    def get_side(tokens: List[str]) -> str:
        """Extract anatomical laterality."""
        if any(t in ["l", "left", "lft", "lt", "左"] for t in tokens):
            return "left"
        if any(t in ["r", "right", "rgt", "rt", "右"] for t in tokens):
            return "right"
        return "center"

    @classmethod
    def match(cls, bone_names: List[str]) -> Dict[str, str]:
        """Calculate score-maximizing 1-to-1 canonical assignment."""
        mapping: Dict[str, str] = {}
        used_bones: Set[str] = set()

        for canonical in cls.CANONICAL_ORDER:
            info = cls.CANONICAL_DEFS[canonical]
            best_bone = None
            best_score = -999.0
            req_side = info["side"]

            for b in bone_names:
                if b in used_bones:
                    continue

                toks = cls.tokenize(b)
                b_side = cls.get_side(toks)

                # Strict lateral isolation
                if req_side == "left" and b_side == "right":
                    continue
                if req_side == "right" and b_side == "left":
                    continue

                keyword_score = 0.0

                # High-priority anatomical keywords
                for k in info.get("high_keys", set()):
                    if k in toks:
                        keyword_score += 70.0
                    elif any(k in t for t in toks):
                        keyword_score += 35.0

                # Secondary keywords
                for k in info.get("keys", set()):
                    if k in toks:
                        keyword_score += 40.0
                    elif any(k in t for t in toks):
                        keyword_score += 20.0

                # Low-confidence keywords
                for k in info.get("low_keys", set()):
                    if k in toks:
                        keyword_score += 10.0

                # Multi-token combination matches (e.g. upper + arm)
                for t1, t2 in info.get("combo", []):
                    if t1 in toks and t2 in toks:
                        keyword_score += 65.0

                # Must match at least one anatomical concept
                if keyword_score == 0.0:
                    continue

                score = keyword_score
                if req_side != "center" and req_side == b_side:
                    score += 25.0
                elif req_side == "center" and b_side == "center":
                    score += 15.0

                # Penalties for disqualifying words
                for bad in info.get("bad", set()):
                    if bad in toks:
                        score -= 80.0

                # Penalties for auxiliary / twist / helper bones
                for aux in ["twist", "share", "sharebone", "nub", "end", "scale", "ctrl", "ik", "socket", "pole"]:
                    if aux in toks:
                        score -= 40.0

                if score > best_score and score > 20.0:
                    best_score = score
                    best_bone = b

            if best_bone:
                mapping[canonical] = best_bone
                used_bones.add(best_bone)

        return mapping


class TopologicalRigSolver:
    """Kinematic graph & 3D spatial geometry solver.
    
    Traces parent-child trees and calculates global rest positions (FK)
    to identify humanoid limbs geometrically when bone names are obscure,
    and measures rest-pose arm angle for auto-calibration.
    """

    @classmethod
    def compute_world_transforms(
        cls,
        nodes: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """Compute world [x, y, z] positions and rotations of nodes via Forward Kinematics."""
        import numpy as np

        def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return np.array([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ], dtype=np.float32)

        def _rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
            q_inv = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
            return _quat_mult(_quat_mult(q, vq), q_inv)[:3]

        world_pos: Dict[int, np.ndarray] = {}
        world_rot: Dict[int, np.ndarray] = {}

        # Find root nodes
        children_set = set()
        for n in nodes:
            children_set.update(n.get("children", []))
        roots = [i for i in range(len(nodes)) if i not in children_set]
        if not roots and nodes:
            roots = [0]

        def _fk(idx: int, p_pos: np.ndarray, p_rot: np.ndarray) -> None:
            n = nodes[idx]
            loc_t = np.array(n.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            loc_r = np.array(n.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

            cur_pos = p_pos + _rotate_vec(p_rot, loc_t)
            cur_rot = _quat_mult(p_rot, loc_r)

            world_pos[idx] = cur_pos
            world_rot[idx] = cur_rot

            for ch in n.get("children", []):
                _fk(ch, cur_pos, cur_rot)

        for r in roots:
            _fk(r, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        return world_pos, world_rot

    @classmethod
    def calculate_rest_arm_angle(
        cls,
        nodes: List[Dict[str, Any]],
        mapping: Dict[str, str],
        world_pos: Dict[int, Any],
    ) -> float:
        """Measure downward inclination angle of the character's upper arm in rest pose."""
        import numpy as np
        name_to_idx = {n.get("name", ""): i for i, n in enumerate(nodes) if "name" in n}

        l_arm = mapping.get("left_arm")
        l_forearm = mapping.get("left_forearm")

        if not l_arm or not l_forearm:
            return 0.0

        idx_arm = name_to_idx.get(l_arm)
        idx_fore = name_to_idx.get(l_forearm)

        if idx_arm is None or idx_fore is None:
            return 0.0
        if idx_arm not in world_pos or idx_fore not in world_pos:
            return 0.0

        p_arm = world_pos[idx_arm]
        p_fore = world_pos[idx_fore]
        vec = p_fore - p_arm

        lateral = abs(float(vec[0]))
        # Detect vertical axis: check if Y or Z has greater vertical variation
        vert = float(vec[1]) if abs(float(vec[1])) >= abs(float(vec[2])) else float(vec[2])

        if lateral < 1e-4:
            return 0.0

        angle_rad = np.arctan2(max(0.0, -vert), lateral)
        return float(np.degrees(angle_rad))

    @classmethod
    def solve_missing_joints(
        cls,
        nodes: List[Dict[str, Any]],
        joint_indices: List[int],
        existing_mapping: Dict[str, str],
        world_pos: Dict[int, Any],
    ) -> Dict[str, str]:
        """Geometrically identify unmapped humanoid joints by walking the kinematic tree."""
        if not joint_indices or not world_pos:
            return existing_mapping

        mapping = dict(existing_mapping)
        mapped_bones = set(mapping.values())

        # Build parent map for nodes
        parent_map: Dict[int, int] = {}
        for p_idx, n in enumerate(nodes):
            for ch in n.get("children", []):
                parent_map[ch] = p_idx

        # Filter joint nodes with positions
        valid_joints = [idx for idx in joint_indices if idx in world_pos]
        if not valid_joints:
            return mapping

        # Determine vertical axis: test Y vs Z span
        y_span = max(world_pos[i][1] for i in valid_joints) - min(world_pos[i][1] for i in valid_joints)
        z_span = max(world_pos[i][2] for i in valid_joints) - min(world_pos[i][2] for i in valid_joints)
        vert_axis = 1 if y_span >= z_span else 2

        # 1. Fallback Head: highest joint
        if "head" not in mapping:
            head_idx = max(valid_joints, key=lambda i: world_pos[i][vert_axis])
            head_name = nodes[head_idx].get("name", f"bone_{head_idx}")
            if head_name not in mapped_bones:
                mapping["head"] = head_name
                mapped_bones.add(head_name)

        # 2. Fallback Feet: lowest joints
        sorted_by_vert = sorted(valid_joints, key=lambda i: world_pos[i][vert_axis])
        if len(sorted_by_vert) >= 2:
            f1, f2 = sorted_by_vert[0], sorted_by_vert[1]
            # Split left vs right laterally (X axis)
            l_foot_idx = f1 if world_pos[f1][0] > world_pos[f2][0] else f2
            r_foot_idx = f2 if l_foot_idx == f1 else f1

            if "left_foot" not in mapping:
                name = nodes[l_foot_idx].get("name", f"bone_{l_foot_idx}")
                if name not in mapped_bones:
                    mapping["left_foot"] = name
                    mapped_bones.add(name)

            if "right_foot" not in mapping:
                name = nodes[r_foot_idx].get("name", f"bone_{r_foot_idx}")
                if name not in mapped_bones:
                    mapping["right_foot"] = name
                    mapped_bones.add(name)

            # Trace up left leg
            cur = parent_map.get(l_foot_idx)
            if cur is not None and "left_leg" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["left_leg"] = name
                    mapped_bones.add(name)
                cur = parent_map.get(cur)
                if cur is not None and "left_upleg" not in mapping:
                    name = nodes[cur].get("name", f"bone_{cur}")
                    if name not in mapped_bones:
                        mapping["left_upleg"] = name
                        mapped_bones.add(name)

            # Trace up right leg
            cur = parent_map.get(r_foot_idx)
            if cur is not None and "right_leg" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["right_leg"] = name
                    mapped_bones.add(name)
                cur = parent_map.get(cur)
                if cur is not None and "right_upleg" not in mapping:
                    name = nodes[cur].get("name", f"bone_{cur}")
                    if name not in mapped_bones:
                        mapping["right_upleg"] = name
                        mapped_bones.add(name)

        # 3. Fallback Hands: extreme lateral joints
        sorted_by_lat = sorted(valid_joints, key=lambda i: world_pos[i][0])
        if len(sorted_by_lat) >= 2:
            r_hand_idx = sorted_by_lat[0]  # Minimum X (right)
            l_hand_idx = sorted_by_lat[-1]  # Maximum X (left)

            # Trace left arm chain
            cur = l_hand_idx
            if "left_hand" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["left_hand"] = name
                    mapped_bones.add(name)
            cur = parent_map.get(cur)
            if cur is not None and "left_forearm" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["left_forearm"] = name
                    mapped_bones.add(name)
                cur = parent_map.get(cur)
                if cur is not None and "left_arm" not in mapping:
                    name = nodes[cur].get("name", f"bone_{cur}")
                    if name not in mapped_bones:
                        mapping["left_arm"] = name
                        mapped_bones.add(name)

            # Trace right arm chain
            cur = r_hand_idx
            if "right_hand" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["right_hand"] = name
                    mapped_bones.add(name)
            cur = parent_map.get(cur)
            if cur is not None and "right_forearm" not in mapping:
                name = nodes[cur].get("name", f"bone_{cur}")
                if name not in mapped_bones:
                    mapping["right_forearm"] = name
                    mapped_bones.add(name)
                cur = parent_map.get(cur)
                if cur is not None and "right_arm" not in mapping:
                    name = nodes[cur].get("name", f"bone_{cur}")
                    if name not in mapped_bones:
                        mapping["right_arm"] = name
                        mapped_bones.add(name)

        return mapping


def _classify_bones(bone_names: List[str]) -> Tuple[RigType, Dict[str, str]]:
    """Match bone names against canonical aliases and determine the rig standard."""
    mapping = UniversalBoneMatcher.match(bone_names)

    mixamo_count = sum(1 for b in bone_names if "mixamorig" in b.lower())
    rigify_count = sum(1 for b in bone_names if "def-" in b.lower())
    cc_count = sum(1 for b in bone_names if "cc_base" in b.lower())
    vrm_names = {
        "hips", "spine", "chest", "neck", "head",
        "leftupperarm", "rightupperarm", "leftlowerarm", "rightlowerarm",
        "leftupperleg", "rightupperleg", "leftfoot", "rightfoot"
    }
    vrm_count = sum(1 for b in bone_names if b.lower() in vrm_names)

    if mixamo_count >= 4:
        rig_type = RigType.MIXAMO
    elif cc_count >= 4:
        rig_type = RigType.CHARACTER_CREATOR
    elif rigify_count >= 4:
        rig_type = RigType.BLENDER_RIGIFY
    elif vrm_count >= 4:
        rig_type = RigType.VRM_HUMANOID
    else:
        rig_type = RigType.GENERIC

    return rig_type, mapping
