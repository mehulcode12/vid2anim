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

    # Collect joint names
    skin = skins[0]
    joint_indices = skin.get("joints", [])
    bone_names: List[str] = []
    for idx in joint_indices:
        if 0 <= idx < len(nodes):
            name = nodes[idx].get("name", f"bone_{idx}")
            bone_names.append(name)

    # Classify rig type and build mapping
    rig_type, mapping = _classify_bones(bone_names)

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


def _classify_bones(bone_names: List[str]) -> Tuple[RigType, Dict[str, str]]:
    """Match bone names against canonical aliases and determine the rig standard."""
    mapping: Dict[str, str] = {}
    lower_to_orig = {b.lower(): b for b in bone_names}

    mixamo_count = sum(1 for b in bone_names if "mixamorig" in b.lower())
    rigify_count = sum(1 for b in bone_names if "def-" in b.lower())
    cc_count = sum(1 for b in bone_names if "cc_base" in b.lower())
    vrm_names = {
        "hips", "spine", "chest", "neck", "head",
        "leftupperarm", "rightupperarm", "leftlowerarm", "rightlowerarm",
        "leftupperleg", "rightupperleg", "leftfoot", "rightfoot"
    }
    vrm_count = sum(1 for b in bone_names if b.lower() in vrm_names)

    # Separate primary structural bones from auxiliary/twist/share bones
    primary_bones = [b for b in bone_names if not any(x in b.lower() for x in ["twist", "sharebone", "nub", "scale"])]
    secondary_bones = [b for b in bone_names if b not in primary_bones]

    for canonical, aliases in JOINT_ALIASES.items():
        found = False
        for alias in aliases:
            # 1. Exact match
            if alias.lower() in lower_to_orig:
                mapping[canonical] = lower_to_orig[alias.lower()]
                found = True
                break
            
            # 2. Substring match against primary (non-twist) bones first
            for b_orig in primary_bones:
                if alias.lower() in b_orig.lower():
                    mapping[canonical] = b_orig
                    found = True
                    break
            if found:
                break

            # 3. Fallback: Substring match against secondary (twist/auxiliary) bones
            for b_orig in secondary_bones:
                if alias.lower() in b_orig.lower():
                    mapping[canonical] = b_orig
                    found = True
                    break
            if found:
                break

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
