"""Synthetic Asset Generator for Testing and CI/CD.

Generates valid binary GLTF 2.0 (.glb) rigged humanoid models, static meshes,
and synthetic motion sequences without external dependencies.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from ..core.motion_extractor import MotionSequence
from ..core.rig_detector import CANONICAL_JOINTS


def create_synthetic_cube_glb(output_path: Path, size: float = 1.8) -> Path:
    """Create a minimal valid binary .glb containing an unrigged static cube."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 8 vertices for a cube centered horizontally, base at Y=0
    h = size
    w = size * 0.3
    d = size * 0.2
    vertices = np.array([
        [-w, 0.0, -d], [w, 0.0, -d], [w, h, -d], [-w, h, -d],
        [-w, 0.0,  d], [w, 0.0,  d], [w, h,  d], [-w, h,  d],
    ], dtype=np.float32)

    # 12 triangles (36 indices)
    indices = np.array([
        0, 2, 1, 0, 3, 2,  # front
        4, 5, 6, 4, 6, 7,  # back
        0, 1, 5, 0, 5, 4,  # bottom
        2, 3, 7, 2, 7, 6,  # top
        0, 4, 7, 0, 7, 3,  # left
        1, 2, 6, 1, 6, 5   # right
    ], dtype=np.uint16)

    vert_bytes = vertices.tobytes()
    idx_bytes = indices.tobytes()
    
    # Pad to 4-byte boundaries
    idx_pad = (4 - len(idx_bytes) % 4) % 4
    bin_bytes = idx_bytes + (b"\x00" * idx_pad) + vert_bytes
    bin_pad = (4 - len(bin_bytes) % 4) % 4
    bin_bytes += b"\x00" * bin_pad

    gltf = {
        "asset": {"version": "2.0", "generator": "vid2anim-synthetic-generator"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "SyntheticMesh"}],
        "meshes": [{
            "name": "CubeMesh",
            "primitives": [{
                "attributes": {"POSITION": 1},
                "indices": 0,
            }]
        }],
        "buffers": [{"byteLength": len(bin_bytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": len(idx_bytes) + idx_pad, "byteLength": len(vert_bytes), "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(indices),
                "type": "SCALAR",
                "min": [0],
                "max": [7],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            }
        ]
    }

    _write_glb_file(output_path, gltf, bin_bytes)
    return output_path


def create_synthetic_humanoid_glb(output_path: Path, rig_prefix: str = "mixamorig:") -> Path:
    """Create a minimal valid binary .glb containing a rigged humanoid character."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal vertices (6 vertices)
    vertices = np.array([
        [0.0, 1.0, 0.0],   # Hips
        [0.0, 1.3, 0.0],   # Spine
        [0.0, 1.7, 0.0],   # Head
        [-0.4, 1.3, 0.0],  # Left Arm
        [0.4, 1.3, 0.0],   # Right Arm
        [0.0, 0.0, 0.0],   # Feet
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 0, 3, 4, 0, 5, 1], dtype=np.uint16)

    # Skinning weights (uint16 indices, float32 weights)
    # Each vertex influenced 100% by its corresponding bone
    joints = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
        [5, 0, 0, 0],
    ], dtype=np.uint16)

    weights = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    bone_names = [
        f"{rig_prefix}Hips",
        f"{rig_prefix}Spine",
        f"{rig_prefix}Head",
        f"{rig_prefix}LeftArm",
        f"{rig_prefix}RightArm",
        f"{rig_prefix}LeftLeg",
    ]

    # Inverse bind matrices (6 identity 4x4 matrices)
    inv_bind = np.tile(np.eye(4, dtype=np.float32), (len(bone_names), 1, 1))

    # Pack binary buffers
    idx_b = indices.tobytes()
    pos_b = vertices.tobytes()
    joint_b = joints.tobytes()
    weight_b = weights.tobytes()
    inv_bind_b = inv_bind.tobytes()

    bin_data = bytearray()
    
    def add_buf(data: bytes) -> tuple[int, int]:
        offset = len(bin_data)
        bin_data.extend(data)
        pad = (4 - len(data) % 4) % 4
        bin_data.extend(b"\x00" * pad)
        return offset, len(data)

    o_idx, l_idx = add_buf(idx_b)
    o_pos, l_pos = add_buf(pos_b)
    o_jnt, l_jnt = add_buf(joint_b)
    o_wgt, l_wgt = add_buf(weight_b)
    o_inv, l_inv = add_buf(inv_bind_b)

    # Nodes: Node 0 = Mesh Root, Nodes 1-6 = Bones
    nodes = [{"name": "MeshNode", "mesh": 0, "skin": 0}]
    bone_node_indices = []
    for i, b_name in enumerate(bone_names):
        bone_idx = len(nodes)
        nodes.append({"name": b_name, "translation": vertices[i].tolist()})
        bone_node_indices.append(bone_idx)

    gltf = {
        "asset": {"version": "2.0", "generator": "vid2anim-synthetic-humanoid"},
        "scene": 0,
        "scenes": [{"nodes": [0] + bone_node_indices}],
        "nodes": nodes,
        "meshes": [{
            "name": "SkinnedMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 1,
                    "JOINTS_0": 2,
                    "WEIGHTS_0": 3,
                },
                "indices": 0,
            }]
        }],
        "skins": [{
            "name": "Armature",
            "inverseBindMatrices": 4,
            "joints": bone_node_indices,
        }],
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": o_idx, "byteLength": l_idx, "target": 34963},
            {"buffer": 0, "byteOffset": o_pos, "byteLength": l_pos, "target": 34962},
            {"buffer": 0, "byteOffset": o_jnt, "byteLength": l_jnt, "target": 34962},
            {"buffer": 0, "byteOffset": o_wgt, "byteLength": l_wgt, "target": 34962},
            {"buffer": 0, "byteOffset": o_inv, "byteLength": l_inv},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
             "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5123, "count": len(joints), "type": "VEC4"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5126, "count": len(weights), "type": "VEC4"},
            {"bufferView": 4, "byteOffset": 0, "componentType": 5126, "count": len(bone_names), "type": "MAT4"},
        ]
    }

    _write_glb_file(output_path, gltf, bytes(bin_data))
    return output_path


def create_synthetic_motion_sequence(frames: int = 30, fps: float = 30.0) -> MotionSequence:
    """Generate a synthetic motion sequence (human waving and swaying)."""
    t = np.linspace(0.0, frames / fps, frames, dtype=np.float32)
    joint_positions: Dict[str, np.ndarray] = {}

    for j_name in list(CANONICAL_JOINTS) + ["left_collar", "right_collar", "left_fingers", "right_fingers"]:
        pos = np.zeros((frames, 3), dtype=np.float32)
        # Base heights
        if "hips" in j_name:
            pos[:, 1] = 1.0 + 0.05 * np.sin(t * 3.0)  # Gentle hip bounce
        elif "spine" in j_name:
            pos[:, 1] = 1.2
        elif "chest" in j_name:
            pos[:, 1] = 1.4
        elif "neck" in j_name:
            pos[:, 1] = 1.55
        elif "head" in j_name:
            pos[:, 1] = 1.7
        elif j_name == "left_collar":
            pos[:, 0] = 0.15
            pos[:, 1] = 1.45
        elif j_name == "left_shoulder":
            pos[:, 0] = 0.35
            pos[:, 1] = 1.45
        elif j_name == "left_arm":
            pos[:, 0] = 0.35
            pos[:, 1] = 1.45 + 0.2 * np.sin(t * 6.0)  # Arm waving
        elif j_name == "left_forearm":
            pos[:, 0] = 0.60
            pos[:, 1] = 1.45 + 0.3 * np.sin(t * 6.0)
        elif j_name == "left_hand":
            pos[:, 0] = 0.80
            pos[:, 1] = 1.45 + 0.35 * np.sin(t * 6.0)
        elif j_name == "left_fingers":
            pos[:, 0] = 0.90
            pos[:, 1] = 1.45 + 0.35 * np.sin(t * 6.0)
        elif j_name == "right_collar":
            pos[:, 0] = -0.15
            pos[:, 1] = 1.45
        elif j_name == "right_shoulder":
            pos[:, 0] = -0.35
            pos[:, 1] = 1.45
        elif j_name == "right_arm":
            pos[:, 0] = -0.35
            pos[:, 1] = 1.45
        elif j_name == "right_forearm":
            pos[:, 0] = -0.60
            pos[:, 1] = 1.45
        elif j_name == "right_hand":
            pos[:, 0] = -0.80
            pos[:, 1] = 1.45
        elif j_name == "right_fingers":
            pos[:, 0] = -0.90
            pos[:, 1] = 1.45
        elif "left_upleg" in j_name or "left_leg" in j_name:
            pos[:, 0] = 0.15
            pos[:, 1] = 0.5
        elif "right_upleg" in j_name or "right_leg" in j_name:
            pos[:, 0] = -0.15
            pos[:, 1] = 0.5
        elif "toe" in j_name:
            pos[:, 1] = 0.00
            pos[:, 2] = 0.15
        elif "foot" in j_name:
            pos[:, 1] = 0.05

        joint_positions[j_name] = pos

    # Ensure anatomical aliases are populated
    joint_positions["pelvis"] = joint_positions["hips"]
    joint_positions["spine2"] = joint_positions.get("spine1", joint_positions["spine"])
    joint_positions["spine3"] = joint_positions["chest"]
    joint_positions["left_hip"] = joint_positions["left_upleg"]
    joint_positions["right_hip"] = joint_positions["right_upleg"]
    joint_positions["left_knee"] = joint_positions["left_leg"]
    joint_positions["right_knee"] = joint_positions["right_leg"]
    joint_positions["left_ankle"] = joint_positions["left_foot"]
    joint_positions["right_ankle"] = joint_positions["right_foot"]
    joint_positions["left_elbow"] = joint_positions["left_forearm"]
    joint_positions["right_elbow"] = joint_positions["right_forearm"]
    joint_positions["left_wrist"] = joint_positions["left_hand"]
    joint_positions["right_wrist"] = joint_positions["right_hand"]

    root_translation = np.zeros((frames, 3), dtype=np.float32)
    root_translation[:, 1] = 1.0 + 0.05 * np.sin(t * 3.0)

    return MotionSequence(
        fps=fps,
        duration_sec=frames / fps,
        frame_count=frames,
        timestamps=t,
        joint_positions=joint_positions,
        root_translation=root_translation,
    )


def _write_glb_file(output_path: Path, gltf_dict: dict, bin_data: bytes) -> None:
    """Helper to assemble a valid binary GLB container."""
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_bytes = bin_data + (b"\x00" * bin_pad)

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(output_path, "wb") as f:
        # GLB Header
        f.write(b"glTF")
        f.write((2).to_bytes(4, "little"))
        f.write(total_len.to_bytes(4, "little"))
        # JSON Chunk
        f.write(len(json_bytes).to_bytes(4, "little"))
        f.write(b"JSON")
        f.write(json_bytes)
        # BIN Chunk
        f.write(len(bin_bytes).to_bytes(4, "little"))
        f.write(b"BIN\x00")
        f.write(bin_bytes)
