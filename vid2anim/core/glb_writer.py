"""glTF 2.0 Binary (.glb) Writer with Skinned Animation Tracks.

Injects baked skeletal animation tracks into user GLB models without losing
textures, materials, vertex attributes, or mesh quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ..utils.math_3d import quat_multiply
from ..utils.progress import ProgressReporter
from .retargeter import RetargetedAnimation
from .rig_detector import RigAnalysis


_GLTF_MAGIC = 0x46546C67  # 'glTF'
_JSON_CHUNK_TYPE = 0x4E4F534A  # 'JSON'
_BIN_CHUNK_TYPE = 0x004E4942  # 'BIN\0'


class GLBAnimationExporter:
    """Injects skeletal animation tracks into a .glb asset."""

    def __init__(self, reporter: Optional[ProgressReporter] = None) -> None:
        self.reporter = reporter or ProgressReporter()

    def export_animated_glb(
        self,
        source_model_path: str | Path,
        animation: RetargetedAnimation,
        rig_info: RigAnalysis,
        output_glb_path: str | Path,
        clip_name: str = "Vid2Anim_Motion",
        replace_existing: bool = True,
    ) -> Path:
        """Inject baked animation into the target GLB model and write out the result."""
        src = Path(source_model_path)
        out = Path(output_glb_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        self.reporter.report("EXPORT", 0.1, f"Packing animated GLB: {out.name}")

        with open(src, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                raise ValueError(f"Source file {src.name} is not a valid binary GLTF (.glb)")

            # Read JSON chunk
            json_len = int.from_bytes(f.read(4), "little")
            json_type = f.read(4)
            json_bytes = f.read(json_len)
            gltf_data = json.loads(json_bytes.decode("utf-8"))

            # Read existing BIN chunk
            existing_bin = b""
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_len = int.from_bytes(chunk_header[:4], "little")
                chunk_type = chunk_header[4:]
                chunk_data = f.read(chunk_len)
                if chunk_type == b"BIN\x00":
                    existing_bin = chunk_data
                    break

        # Map bone names to glTF node indices and bind rotations
        node_name_to_idx: Dict[str, int] = {}
        node_bind_rotations: Dict[str, np.ndarray] = {}
        for idx, node in enumerate(gltf_data.get("nodes", [])):
            if "name" in node:
                name = node["name"]
                node_name_to_idx[name] = idx
                if "rotation" in node:
                    node_bind_rotations[name] = np.array(node["rotation"], dtype=np.float32)

        # Prepare animation buffer data
        bin_appended = bytearray(existing_bin)
        buffer_idx = 0  # GLB primary buffer

        # Ensure bufferViews and accessors exist
        if "bufferViews" not in gltf_data:
            gltf_data["bufferViews"] = []
        if "accessors" not in gltf_data:
            gltf_data["accessors"] = []
        if "animations" not in gltf_data:
            gltf_data["animations"] = []

        # 1. Write timestamps accessor (float32, SCALAR)
        times_bytes = animation.timestamps.astype(np.float32).tobytes()
        time_offset = len(bin_appended)
        bin_appended.extend(times_bytes)
        _pad_4bytes(bin_appended)

        time_bview_idx = len(gltf_data["bufferViews"])
        gltf_data["bufferViews"].append({
            "buffer": buffer_idx,
            "byteOffset": time_offset,
            "byteLength": len(times_bytes),
        })

        time_accessor_idx = len(gltf_data["accessors"])
        gltf_data["accessors"].append({
            "bufferView": time_bview_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": animation.frame_count,
            "type": "SCALAR",
            "min": [float(animation.timestamps[0])],
            "max": [float(animation.timestamps[-1])],
        })

        # 2. Add samplers and channels for each animated bone
        channels: List[Dict[str, Any]] = []
        samplers: List[Dict[str, Any]] = []

        for bone_name, quat_frames in animation.bone_rotations.items():
            if bone_name not in node_name_to_idx:
                continue

            node_idx = node_name_to_idx[bone_name]

            # If node has a non-identity bind rotation, compose: Q_track = Q_bind * Q_anim
            if bone_name in node_bind_rotations:
                bind_q = node_bind_rotations[bone_name]
                if not np.allclose(bind_q, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
                    composed = np.zeros_like(quat_frames)
                    for f_idx in range(len(quat_frames)):
                        composed[f_idx] = quat_multiply(bind_q, quat_frames[f_idx])
                    quat_frames = composed

            # Write rotation data (float32, VEC4)
            rot_bytes = quat_frames.astype(np.float32).tobytes()
            rot_offset = len(bin_appended)
            bin_appended.extend(rot_bytes)
            _pad_4bytes(bin_appended)

            rot_bview_idx = len(gltf_data["bufferViews"])
            gltf_data["bufferViews"].append({
                "buffer": buffer_idx,
                "byteOffset": rot_offset,
                "byteLength": len(rot_bytes),
            })

            rot_accessor_idx = len(gltf_data["accessors"])
            gltf_data["accessors"].append({
                "bufferView": rot_bview_idx,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": animation.frame_count,
                "type": "VEC4",
            })

            sampler_idx = len(samplers)
            samplers.append({
                "input": time_accessor_idx,
                "interpolation": "LINEAR",
                "output": rot_accessor_idx,
            })

            channels.append({
                "sampler": sampler_idx,
                "target": {
                    "node": node_idx,
                    "path": "rotation",
                },
            })

        # Add root hip translation track if mapped
        root_bone = rig_info.bone_mapping.get("hips")
        if root_bone and root_bone in node_name_to_idx and len(animation.root_translations) > 0:
            root_node_idx = node_name_to_idx[root_bone]
            root_node = gltf_data["nodes"][root_node_idx]
            bind_trans = np.array(root_node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            final_trans = animation.root_translations + bind_trans
            trans_bytes = final_trans.astype(np.float32).tobytes()
            trans_offset = len(bin_appended)
            bin_appended.extend(trans_bytes)
            _pad_4bytes(bin_appended)

            trans_bview_idx = len(gltf_data["bufferViews"])
            gltf_data["bufferViews"].append({
                "buffer": buffer_idx,
                "byteOffset": trans_offset,
                "byteLength": len(trans_bytes),
            })

            trans_accessor_idx = len(gltf_data["accessors"])
            gltf_data["accessors"].append({
                "bufferView": trans_bview_idx,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": animation.frame_count,
                "type": "VEC3",
            })

            trans_sampler_idx = len(samplers)
            samplers.append({
                "input": time_accessor_idx,
                "interpolation": "LINEAR",
                "output": trans_accessor_idx,
            })

            channels.append({
                "sampler": trans_sampler_idx,
                "target": {
                    "node": root_node_idx,
                    "path": "translation",
                },
            })

        new_clip = {
            "name": clip_name,
            "channels": channels,
            "samplers": samplers,
        }
        if replace_existing:
            # Overwrite existing animations so the newly retargeted motion is the sole and active track.
            # This prevents pre-existing animations (e.g. from Sketchfab, Mixamo, or CC) from masking
            # the retargeted motion in 3D viewers and engines.
            gltf_data["animations"] = [new_clip]
        else:
            # Prepend at index 0 so Vid2Anim_Motion is always the default active track
            gltf_data["animations"].insert(0, new_clip)

        # Update primary buffer size in JSON
        if "buffers" in gltf_data and len(gltf_data["buffers"]) > 0:
            gltf_data["buffers"][0]["byteLength"] = len(bin_appended)

        # Write new binary GLB
        new_json_bytes = json.dumps(gltf_data, separators=(",", ":")).encode("utf-8")
        _pad_4bytes(new_json_bytes)

        json_len = len(new_json_bytes)
        bin_len = len(bin_appended)
        total_len = 12 + 8 + json_len + 8 + bin_len

        with open(out, "wb") as f_out:
            # GLB Header
            f_out.write(b"glTF")
            f_out.write((2).to_bytes(4, "little"))
            f_out.write(total_len.to_bytes(4, "little"))

            # JSON Chunk
            f_out.write(json_len.to_bytes(4, "little"))
            f_out.write(b"JSON")
            f_out.write(new_json_bytes)

            # BIN Chunk
            f_out.write(bin_len.to_bytes(4, "little"))
            f_out.write(b"BIN\x00")
            f_out.write(bin_appended)

        self.reporter.report("EXPORT", 1.0, f"Successfully created: {out.name} ({total_len / (1024*1024):.2f} MB)")
        return out


def _pad_4bytes(buffer: bytearray | bytes) -> None:
    """Pad buffer with zeros or spaces to 4-byte alignment."""
    rem = len(buffer) % 4
    if rem != 0:
        if isinstance(buffer, bytearray):
            buffer.extend(b"\x00" * (4 - rem))
