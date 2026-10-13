"""Rigorous Edge-Case and Boundary Condition Test Suite for vid2anim."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from vid2anim.core.auto_rig import fit_skeleton_to_mesh
from vid2anim.core.glb_writer import GLBAnimationExporter
from vid2anim.core.interpolation import sanitize_and_interpolate_motion
from vid2anim.core.motion_extractor import MotionSequence
from vid2anim.core.retargeter import UniversalRetargeter
from vid2anim.core.rig_detector import RigType, _classify_bones, inspect_model, parse_glb_json
from vid2anim.core.scale_normalizer import ScaleNormalizer
from vid2anim.core.smoothing import OneEuroFilter1D, TemporalMotionSmoother
from vid2anim.pipeline import PipelineConfig, Vid2AnimPipeline
from vid2anim.utils.fixtures import (
    create_synthetic_cube_glb,
    create_synthetic_humanoid_glb,
    create_synthetic_motion_sequence,
)
from vid2anim.utils.math_3d import (
    continue_quaternion,
    decompose_swing_twist,
    normalize,
    quat_conjugate,
    quat_identity,
    quat_multiply,
    quat_rotate_vector,
    quat_slerp,
    rotation_between_vectors,
)


# =====================================================================
# 1. Math Edge Cases
# =====================================================================

def test_math_zero_vector_normalization():
    """Zero vectors must not cause divide-by-zero or NaNs."""
    zero_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    res = normalize(zero_v)
    assert np.all(res == 0.0)
    assert not np.isnan(res).any()


def test_math_antiparallel_rotation_singularity():
    """Vectors pointing in exact opposite directions (dot = -1.0) must produce valid orthogonal swing."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    q = rotation_between_vectors(v1, v2)

    assert not np.isnan(q).any()
    assert np.isclose(np.linalg.norm(q), 1.0)
    # Rotating v1 by q should yield v2
    rotated = quat_rotate_vector(q, v1)
    assert np.allclose(rotated, v2, atol=1e-5)


def test_math_tiny_angle_rotation():
    """Vectors with microscopic angle difference must return identity without NaN."""
    v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0 + 1e-9, 0.0], dtype=np.float32)
    q = rotation_between_vectors(v1, v2)
    assert not np.isnan(q).any()
    assert np.allclose(q, quat_identity())


def test_math_slerp_boundary_conditions():
    """SLERP at t=0 and t=1 must yield exact endpoints."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q1 = np.array([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float32)

    s0 = quat_slerp(q0, q1, 0.0)
    assert np.allclose(s0, q0)

    s1 = quat_slerp(q0, q1, 1.0)
    assert np.allclose(s1, q1)

    # Opposing hemisphere quaternions (q and -q) must interpolate smoothly
    s_mid = quat_slerp(q0, -q0, 0.5)
    assert not np.isnan(s_mid).any()


def test_math_swing_twist_decomposition():
    """Decomposing pure twist around Y axis."""
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # 90 deg rotation around Y is pure twist
    q_twist_in = np.array([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float32)
    q_swing, q_twist = decompose_swing_twist(q_twist_in, y_axis)

    assert np.allclose(q_swing, quat_identity(), atol=1e-4)
    assert np.allclose(q_twist, q_twist_in, atol=1e-4)


# =====================================================================
# 2. Tracking Anomalies & Outlier Sanitization
# =====================================================================

def test_interpolation_nan_and_inf_recovery():
    """Missing tracking frames containing NaN or Inf must be reconstructed seamlessly."""
    frames = 10
    pos = np.zeros((frames, 3), dtype=np.float32)
    # True trajectory is linear from 0 to 10
    for i in range(frames):
        pos[i, 1] = float(i)

    # Corrupt frames 3 and 4 with NaN and Inf
    pos[3, 1] = np.nan
    pos[4, 1] = np.inf

    raw_motion = MotionSequence(
        fps=30.0,
        duration_sec=0.33,
        frame_count=frames,
        timestamps=np.linspace(0, 0.33, frames, dtype=np.float32),
        joint_positions={"left_hand": pos},
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )

    cleaned = sanitize_and_interpolate_motion(raw_motion)
    cleaned_y = cleaned.joint_positions["left_hand"][:, 1]

    assert not np.isnan(cleaned_y).any()
    assert not np.isinf(cleaned_y).any()
    # Interpolated values should be roughly 3.0 and 4.0
    assert np.isclose(cleaned_y[3], 3.0, atol=0.1)
    assert np.isclose(cleaned_y[4], 4.0, atol=0.1)


def test_interpolation_velocity_spike_clamping():
    """Sudden single-frame teleportation due to tracking glitch must be clamped."""
    frames = 5
    pos = np.zeros((frames, 3), dtype=np.float32)
    # Sudden jump of 10 meters at frame 2
    pos[2] = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    raw_motion = MotionSequence(
        fps=30.0,
        duration_sec=0.16,
        frame_count=frames,
        timestamps=np.linspace(0, 0.16, frames, dtype=np.float32),
        joint_positions={"head": pos},
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )

    cleaned = sanitize_and_interpolate_motion(raw_motion, max_velocity_mps=5.0)
    # The glitch at frame 2 should be replaced by interpolation
    assert np.linalg.norm(cleaned.joint_positions["head"][2]) < 2.0


# =====================================================================
# 3. Model Rig Detection & Formats Edge Cases
# =====================================================================

def test_bone_classification_vrm_and_rigify():
    """VRM / Unity Humanoid and Blender Rigify naming standards."""
    vrm_bones = ["hips", "spine", "chest", "neck", "head", "leftUpperArm", "rightUpperArm"]
    r_type, mapping = _classify_bones(vrm_bones)
    assert r_type == RigType.VRM_HUMANOID
    assert mapping["left_arm"] == "leftUpperArm"

    rigify_bones = ["def-spine", "def-chest", "def-neck", "def-head", "def-upper_arm.l", "def-thigh.l"]
    r_type, mapping = _classify_bones(rigify_bones)
    assert r_type == RigType.BLENDER_RIGIFY
    assert mapping["left_arm"] == "def-upper_arm.l"


def test_synthetic_cube_glb_inspection():
    """Synthetic unrigged cube GLB should report is_rigged=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test_cube.glb"
        create_synthetic_cube_glb(glb_path)

        analysis = inspect_model(glb_path)
        assert not analysis.is_rigged
        assert analysis.rig_type == RigType.UNRIGGED
        assert analysis.vertex_count == 8


def test_synthetic_humanoid_glb_inspection():
    """Synthetic rigged humanoid GLB should report is_rigged=True and Mixamo profile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test_humanoid.glb"
        create_synthetic_humanoid_glb(glb_path, rig_prefix="mixamorig:")

        analysis = inspect_model(glb_path)
        assert analysis.is_rigged
        assert analysis.rig_type == RigType.MIXAMO
        assert "hips" in analysis.bone_mapping
        assert "spine" in analysis.bone_mapping


# =====================================================================
# 4. GLB Binary Container Integrity & Animation Tracks
# =====================================================================

def test_glb_binary_container_integrity():
    """Verify that exporting animations preserves binary glTF 2.0 specifications."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source_glb = tmp / "character.glb"
        out_glb = tmp / "character_animated.glb"

        create_synthetic_humanoid_glb(source_glb)
        rig_info = inspect_model(source_glb)

        motion = create_synthetic_motion_sequence(frames=15, fps=30.0)
        retargeter = UniversalRetargeter()
        anim = retargeter.retarget(motion, rig_info)

        exporter = GLBAnimationExporter()
        exporter.export_animated_glb(source_glb, anim, rig_info, out_glb)

        assert out_glb.exists()

        # Verify low-level binary GLB specification
        with open(out_glb, "rb") as f:
            magic = f.read(4)
            assert magic == b"glTF", "Magic bytes must be 'glTF'"
            version = int.from_bytes(f.read(4), "little")
            assert version == 2, "glTF version must be 2"
            file_len = int.from_bytes(f.read(4), "little")
            assert file_len == out_glb.stat().st_size, "Header length must equal total file size"

            # Chunk 0: JSON
            json_len = int.from_bytes(f.read(4), "little")
            json_type = f.read(4)
            assert json_type == b"JSON"
            json_bytes = f.read(json_len)
            gltf_json = json.loads(json_bytes.decode("utf-8"))

            # Chunk 1: BIN
            bin_len = int.from_bytes(f.read(4), "little")
            bin_type = f.read(4)
            assert bin_type == b"BIN\x00"

        # Check animations block
        assert "animations" in gltf_json
        assert len(gltf_json["animations"]) >= 1
        clip = gltf_json["animations"][0]
        assert clip["name"] == "Vid2Anim_Motion"
        assert len(clip["channels"]) > 0


# =====================================================================
# 5. Full End-to-End Pipeline on Synthetic Inputs
# =====================================================================

def test_full_pipeline_on_synthetic_humanoid():
    """Run Vid2AnimPipeline end-to-end on synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create dummy image without requiring external packages
        media_path = tmp / "dummy_photo.bmp"
        # 1x1 24-bit BMP image (54 bytes)
        bmp_header = (
            b"BM"  # Signature
            + (58).to_bytes(4, "little")  # File size
            + (0).to_bytes(4, "little")  # Reserved
            + (54).to_bytes(4, "little")  # Offset
            + (40).to_bytes(4, "little")  # Header size
            + (1).to_bytes(4, "little")  # Width
            + (1).to_bytes(4, "little")  # Height
            + (1).to_bytes(2, "little")  # Planes
            + (24).to_bytes(2, "little")  # Bit count
            + (0).to_bytes(4, "little")  # Compression
            + (4).to_bytes(4, "little")  # Image size
            + (0).to_bytes(4, "little")  # X pixels per meter
            + (0).to_bytes(4, "little")  # Y pixels per meter
            + (0).to_bytes(4, "little")  # Colors used
            + (0).to_bytes(4, "little")  # Important colors
            + b"\x80\x80\x80\x00"  # 1 pixel + padding
        )
        media_path.write_bytes(bmp_header)

        model_path = tmp / "model.glb"
        out_glb = tmp / "output_animated.glb"
        out_bvh = tmp / "output.bvh"

        # Create synthetic rigged GLB
        create_synthetic_humanoid_glb(model_path)

        config = PipelineConfig(
            input_media=media_path,
            target_model=model_path,
            output_glb=out_glb,
            output_bvh=out_bvh,
            device="cpu",
            max_duration_sec=30.0,
            smooth=True,
        )

        pipeline = Vid2AnimPipeline(config)
        result = pipeline.execute()

        assert result.success, f"Pipeline execution failed: {result.error_message}"
        assert out_glb.exists()
        assert out_bvh.exists()
        assert result.frame_count >= 1
        assert result.elapsed_time_sec >= 0.0
