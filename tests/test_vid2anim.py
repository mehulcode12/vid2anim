"""Basic test suite for vid2anim core modules."""

import tempfile
from pathlib import Path
import numpy as np

from vid2anim.core.auto_rig import fit_skeleton_to_mesh
from vid2anim.core.motion_extractor import MotionSequence
from vid2anim.core.rig_detector import _classify_bones
from vid2anim.utils.diagnostics import get_system_diagnostics
from vid2anim.utils.math_3d import (
    continue_quaternion,
    normalize,
    quat_identity,
    quat_multiply,
    rotation_between_vectors,
)


def test_math_quaternions():
    # Test normalization
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    v_norm = normalize(v)
    assert np.isclose(np.linalg.norm(v_norm), 1.0)

    # Test rotation between identical vectors
    q_id = rotation_between_vectors(v_norm, v_norm)
    assert np.allclose(q_id, quat_identity())

    # Test 90 degree rotation
    v_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    q_xy = rotation_between_vectors(v_x, v_y)
    assert np.isclose(np.linalg.norm(q_xy), 1.0)

    # Test continue_quaternion
    q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q_neg = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    q_fixed = continue_quaternion(q_neg, q1)
    assert np.allclose(q_fixed, q1)


def test_diagnostics():
    diag = get_system_diagnostics()
    assert "os" in diag
    assert "cpu_count" in diag
    assert diag["cpu_count"] >= 1
    assert "cuda_available" in diag
    assert diag["recommended_device"] in ["cpu", "cuda"]


def test_bone_classification():
    mixamo_bones = [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Neck",
        "mixamorig:Head",
        "mixamorig:LeftArm",
        "mixamorig:RightArm",
    ]
    rig_type, mapping = _classify_bones(mixamo_bones)
    assert rig_type.value == "mixamo"
    assert "hips" in mapping
    assert mapping["hips"] == "mixamorig:Hips"


def test_auto_rigging():
    # Synthetic cube vertices
    verts = np.array([
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [0.5, 1.8, -0.5],
        [-0.5, 1.8, -0.5],
        [-0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 1.8, 0.5],
        [-0.5, 1.8, 0.5],
    ], dtype=np.float32)

    auto_rig = fit_skeleton_to_mesh(verts)
    assert len(auto_rig.joint_positions) == 19
    assert auto_rig.vertex_joints.shape == (8, 4)
    assert auto_rig.vertex_weights.shape == (8, 4)
    # Weights should sum to 1.0
    weight_sums = np.sum(auto_rig.vertex_weights, axis=-1)
    assert np.allclose(weight_sums, 1.0)


def test_motion_cache_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "motion_test.npz"
        motion = MotionSequence(
            fps=30.0,
            duration_sec=1.0,
            frame_count=30,
            timestamps=np.linspace(0.0, 1.0, 30, dtype=np.float32),
            joint_positions={"hips": np.zeros((30, 3), dtype=np.float32)},
            root_translation=np.zeros((30, 3), dtype=np.float32),
        )
        motion.save_cache(tmp_path)
        assert tmp_path.exists()

        loaded = MotionSequence.load_cache(tmp_path)
        assert loaded.fps == 30.0
        assert loaded.frame_count == 30
        assert "hips" in loaded.joint_positions


def test_temporal_smoothing():
    from vid2anim.core.smoothing import TemporalMotionSmoother

    # Create noisy 10-frame motion
    t = np.linspace(0, 1, 10, dtype=np.float32)
    noisy_hips = np.zeros((10, 3), dtype=np.float32)
    noisy_hips[:, 1] = 1.0 + np.sin(t * 10) * 0.1  # sine variation
    motion = MotionSequence(
        fps=30.0,
        duration_sec=0.33,
        frame_count=10,
        timestamps=t,
        joint_positions={"hips": noisy_hips},
        root_translation=noisy_hips,
    )

    smoother = TemporalMotionSmoother()
    smoothed = smoother.smooth_motion_positions(motion)
    assert smoothed.frame_count == 10
    # Smoothed trajectory should be finite and close to original
    assert np.all(np.isfinite(smoothed.joint_positions["hips"]))


def test_pose_calibration():
    from vid2anim.core.calibration import PoseCalibrator, RestPoseType

    t_calib = PoseCalibrator(RestPoseType.T_POSE)
    offset_t = t_calib.get_rest_offset("left_arm")
    assert np.allclose(offset_t, [0, 0, 0, 1])

    a_calib = PoseCalibrator(RestPoseType.A_POSE_30)
    offset_a = a_calib.get_rest_offset("left_arm")
    # Should have non-zero rotation for arm
    assert not np.allclose(offset_a, [0, 0, 0, 1])


def test_pipeline_validation():
    from vid2anim.pipeline import PipelineConfig, Vid2AnimPipeline

    config = PipelineConfig(
        input_media=Path("non_existent_file.mp4"),
        target_model=Path("non_existent_model.glb"),
    )
    pipeline = Vid2AnimPipeline(config)
    result = pipeline.execute()
    assert not result.success
    assert "does not exist" in result.error_message.lower()


def test_dual_engine_motion_extractor():
    from vid2anim.core.motion_extractor import MotionExtractor

    # 1. Test engine=mediapipe initialization
    ext_mp = MotionExtractor(engine="mediapipe")
    assert ext_mp.engine == "mediapipe"

    # 2. Test engine=dwpose initialization
    ext_dw = MotionExtractor(engine="dwpose")
    assert ext_dw.engine == "dwpose"

    # 3. Test DWPose landmark mapping logic
    dummy_kpts = np.zeros((133, 2), dtype=np.float32)
    # Put hips at y=400, shoulders at y=200
    dummy_kpts[11] = [300, 400]  # left hip
    dummy_kpts[12] = [300, 400]  # right hip
    dummy_kpts[5] = [280, 200]   # left shoulder
    dummy_kpts[6] = [320, 200]   # right shoulder
    dummy_kpts[99] = [260, 160]  # left forefinger tip
    dummy_kpts[120] = [340, 160] # right forefinger tip
    dummy_scores = np.ones(133, dtype=np.float32) * 0.9

    mapped = ext_dw._map_dwpose_landmarks(dummy_kpts, dummy_scores, 640, 480)
    assert "hips" in mapped
    assert "left_fingers" in mapped
    assert "right_fingers" in mapped
    assert "left_middle" in mapped
    assert "right_middle" in mapped
    assert "left_thumb" in mapped
    assert "right_thumb" in mapped
    assert mapped["hips"].shape == (3,)
    assert mapped["left_fingers"].shape == (3,)


def test_top_level_package_exports():
    import vid2anim

    assert hasattr(vid2anim, "__version__")
    assert vid2anim.__version__ == "0.1.0"
    for symbol in vid2anim.__all__:
        assert hasattr(vid2anim, symbol), f"Symbol {symbol} missing from vid2anim module"


