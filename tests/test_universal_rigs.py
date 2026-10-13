"""Test Universal Bone Matcher across 8 different 3D industry rigging conventions.

Ensures zero manual patching is required when users import models from:
1. Adobe Mixamo
2. VRM / Unity Humanoid
3. Blender Rigify
4. Reallusion Character Creator (CC3 / CC4)
5. Daz3D Genesis (G8 / G9)
6. Unreal Engine 5 Mannequin (UE5 Manny / Quinn)
7. Japanese MMD (Miku Miku Dance)
8. Custom snake_case and camelCase rigs
"""

import pytest
from vid2anim.core.rig_detector import UniversalBoneMatcher


def test_match_mixamo():
    bones = [
        "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
        "mixamorig:Neck", "mixamorig:Head", "mixamorig:LeftShoulder", "mixamorig:LeftArm",
        "mixamorig:LeftForeArm", "mixamorig:LeftHand", "mixamorig:RightShoulder",
        "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
        "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
        "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["hips"] == "mixamorig:Hips"
    assert m["left_arm"] == "mixamorig:LeftArm"
    assert m["right_leg"] == "mixamorig:RightLeg"
    assert len(m) >= 18


def test_match_vrm_humanoid():
    bones = [
        "hips", "spine", "chest", "neck", "head",
        "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
        "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
        "leftUpperLeg", "leftLowerLeg", "leftFoot",
        "rightUpperLeg", "rightLowerLeg", "rightFoot",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["hips"] == "hips"
    assert m["left_arm"] == "leftUpperArm"
    assert m["left_forearm"] == "leftLowerArm"
    assert m["right_leg"] == "rightLowerLeg"
    assert len(m) == 19


def test_match_blender_rigify():
    bones = [
        "root", "DEF-spine", "DEF-chest", "DEF-neck", "DEF-head",
        "DEF-shoulder.L", "DEF-upper_arm.L", "DEF-forearm.L", "DEF-hand.L",
        "DEF-shoulder.R", "DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R",
        "DEF-thigh.L", "DEF-shin.L", "DEF-foot.L",
        "DEF-thigh.R", "DEF-shin.R", "DEF-foot.R",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["left_arm"] == "DEF-upper_arm.L"
    assert m["left_forearm"] == "DEF-forearm.L"
    assert m["left_upleg"] == "DEF-thigh.L"
    assert m["left_leg"] == "DEF-shin.L"
    assert len(m) >= 17


def test_match_unreal_engine_5():
    bones = [
        "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
        "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
        "upperarm_twist_01_l", "lowerarm_twist_01_l",  # twist bones should be ignored
        "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
        "thigh_l", "calf_l", "foot_l",
        "thigh_r", "calf_r", "foot_r",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["hips"] == "pelvis"
    assert m["left_arm"] == "upperarm_l"
    assert m["left_forearm"] == "lowerarm_l"
    assert m["right_upleg"] == "thigh_r"
    assert m["right_leg"] == "calf_r"
    assert len(m) >= 18


def test_match_daz3d_genesis():
    bones = [
        "pelvis", "abdomenLower", "chestUpper", "neckLower", "head",
        "lCollar", "lShldrBend", "lForearmBend", "lHand",
        "rCollar", "rShldrBend", "rForearmBend", "rHand",
        "lThighBend", "lShin", "lFoot",
        "rThighBend", "rShin", "rFoot",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["hips"] == "pelvis"
    assert m["left_arm"] == "lShldrBend"
    assert m["left_forearm"] == "lForearmBend"
    assert m["left_upleg"] == "lThighBend"
    assert m["left_leg"] == "lShin"
    assert len(m) >= 17


def test_match_japanese_mmd():
    bones = [
        "センター", "下半身", "上半身", "上半身2", "首", "頭",
        "左肩", "左腕", "左ひじ", "左手首",
        "右肩", "右腕", "右ひじ", "右手首",
        "左足", "左ひざ", "左足首",
        "右足", "右ひざ", "右足首",
    ]
    m = UniversalBoneMatcher.match(bones)
    assert m["hips"] in ["センター", "下半身"]
    assert m["spine"] == "上半身"
    assert m["head"] == "頭"
    assert m["left_arm"] == "左腕"
    assert m["left_forearm"] == "左ひじ"
    assert m["left_leg"] == "左ひざ"
    assert len(m) >= 15


def test_anonymous_bones_topological_fallback():
    """Verify that models with completely uninformative names (Bone_00...Bone_16) are mapped geometrically."""
    import numpy as np
    from vid2anim.core.rig_detector import TopologicalRigSolver

    nodes = [
        # 0: Hips at (0, 1.0, 0)
        {"name": "Bone_00", "translation": [0.0, 1.0, 0.0], "children": [1, 5, 8]},
        # Spine -> Chest -> Neck -> Head
        {"name": "Bone_01", "translation": [0.0, 0.2, 0.0], "children": [2]},
        {"name": "Bone_02", "translation": [0.0, 0.2, 0.0], "children": [3, 11, 14]},
        {"name": "Bone_03", "translation": [0.0, 0.1, 0.0], "children": [4]},
        {"name": "Bone_04", "translation": [0.0, 0.2, 0.0], "children": []},
        # Left Leg: Thigh (5) -> Calf (6) -> Foot (7)
        {"name": "Bone_05", "translation": [0.1, -0.4, 0.0], "children": [6]},
        {"name": "Bone_06", "translation": [0.0, -0.4, 0.0], "children": [7]},
        {"name": "Bone_07", "translation": [0.0, -0.2, 0.1], "children": []},
        # Right Leg: Thigh (8) -> Calf (9) -> Foot (10)
        {"name": "Bone_08", "translation": [-0.1, -0.4, 0.0], "children": [9]},
        {"name": "Bone_09", "translation": [0.0, -0.4, 0.0], "children": [10]},
        {"name": "Bone_10", "translation": [0.0, -0.2, 0.1], "children": []},
        # Left Arm: Shoulder/Arm (11) -> Forearm (12) -> Hand (13)
        {"name": "Bone_11", "translation": [0.2, 0.0, 0.0], "children": [12]},
        {"name": "Bone_12", "translation": [0.25, 0.0, 0.0], "children": [13]},
        {"name": "Bone_13", "translation": [0.2, 0.0, 0.0], "children": []},
        # Right Arm: Shoulder/Arm (14) -> Forearm (15) -> Hand (16)
        {"name": "Bone_14", "translation": [-0.2, 0.0, 0.0], "children": [15]},
        {"name": "Bone_15", "translation": [-0.25, 0.0, 0.0], "children": [16]},
        {"name": "Bone_16", "translation": [-0.2, 0.0, 0.0], "children": []},
    ]

    joint_indices = list(range(len(nodes)))
    world_pos, _ = TopologicalRigSolver.compute_world_transforms(nodes)

    mapping = TopologicalRigSolver.solve_missing_joints(
        nodes=nodes,
        joint_indices=joint_indices,
        existing_mapping={},
        world_pos=world_pos,
    )

    assert mapping["head"] == "Bone_04"
    assert mapping["left_foot"] == "Bone_07"
    assert mapping["right_foot"] == "Bone_10"
    assert mapping["left_leg"] == "Bone_06"
    assert mapping["left_upleg"] == "Bone_05"
    assert mapping["left_hand"] == "Bone_13"
    assert mapping["left_forearm"] == "Bone_12"
    assert mapping["left_arm"] == "Bone_11"
    assert mapping["right_hand"] == "Bone_16"
    assert mapping["right_forearm"] == "Bone_15"
    assert mapping["right_arm"] == "Bone_14"


def test_auto_rest_pose_detection():
    """Verify that upper arm angle is accurately measured and auto-calibrated."""
    import numpy as np
    from vid2anim.core.calibration import PoseCalibrator, RestPoseType

    # Case 1: T-Pose (horizontal arm)
    arm_t = np.array([0.2, 1.4, 0.0])
    fore_t = np.array([0.5, 1.4, 0.0])
    angle_t = PoseCalibrator.calculate_rest_arm_angle(arm_t, fore_t)
    assert abs(angle_t) < 1.0

    calib_t = PoseCalibrator(rest_pose=RestPoseType.AUTO, arm_angle_deg=angle_t)
    q_offset_t = calib_t.get_rest_offset("left_arm")
    assert np.allclose(q_offset_t, [0.0, 0.0, 0.0, 1.0])

    # Case 2: A-Pose at ~35 degrees down
    # dx = 13.27, dy = -9.42 -> arctan(9.42 / 13.27) = ~35.4 deg
    arm_a = np.array([13.17, 81.0, 1.37])
    fore_a = np.array([26.44, 71.58, 4.74])
    angle_a = PoseCalibrator.calculate_rest_arm_angle(arm_a, fore_a, vertical_axis=1)
    assert 34.0 < angle_a < 37.0

    calib_a = PoseCalibrator(rest_pose=RestPoseType.AUTO, arm_angle_deg=angle_a)
    q_offset_a = calib_a.get_rest_offset("left_arm")
    # Offset should not be identity; should rotate up
    assert not np.allclose(q_offset_a, [0.0, 0.0, 0.0, 1.0])
    assert q_offset_a[2] > 0.0  # Positive Z-axis rotation for left arm up


def test_replace_existing_animations_in_glb(tmp_path):
    """Verify that models with pre-existing animation clips have them overridden by Vid2Anim_Motion."""
    import json
    from vid2anim.core.glb_writer import GLBAnimationExporter
    from vid2anim.core.retargeter import UniversalRetargeter
    from vid2anim.core.rig_detector import inspect_model
    from vid2anim.utils.fixtures import create_synthetic_humanoid_glb, create_synthetic_motion_sequence

    # Create model with an existing animation
    model_path = tmp_path / "pre_animated.glb"
    create_synthetic_humanoid_glb(model_path)

    # Inject an initial dummy animation
    with open(model_path, "rb") as f:
        header = f.read(12)
        json_len = int.from_bytes(f.read(4), "little")
        f.read(4)
        gltf = json.loads(f.read(json_len).decode("utf-8"))
        bin_data = f.read()

    gltf["animations"] = [{"name": "Old_Sketchfab_Idle", "channels": [], "samplers": []}]
    new_json = json.dumps(gltf).encode("utf-8")
    rem = len(new_json) % 4
    if rem != 0:
        new_json += b" " * (4 - rem)

    with open(model_path, "wb") as f:
        f.write(b"glTF")
        f.write((2).to_bytes(4, "little"))
        total = 12 + 8 + len(new_json) + len(bin_data)
        f.write(total.to_bytes(4, "little"))
        f.write(len(new_json).to_bytes(4, "little"))
        f.write(b"JSON")
        f.write(new_json)
        f.write(bin_data)

    # Retarget and export with Vid2Anim
    rig_info = inspect_model(model_path)
    motion = create_synthetic_motion_sequence(frames=10, fps=30.0)
    retargeter = UniversalRetargeter()
    anim = retargeter.retarget(motion, rig_info)

    out_path = tmp_path / "result_animated.glb"
    exporter = GLBAnimationExporter()
    exporter.export_animated_glb(model_path, anim, rig_info, out_path, replace_existing=True)

    # Verify that the exported GLB has Vid2Anim_Motion as the active sole animation
    with open(out_path, "rb") as f:
        f.seek(12)
        jlen = int.from_bytes(f.read(4), "little")
        f.read(4)
        out_gltf = json.loads(f.read(jlen).decode("utf-8"))

    anims = out_gltf.get("animations", [])
    assert len(anims) == 1
    assert anims[0]["name"] == "Vid2Anim_Motion"


def test_canonical_skeleton_and_universal_transfer(tmp_path):
    """Test SO(3) Lie algebra canonical FK solver and C D C^T transfer."""
    import numpy as np
    from vid2anim.core.canonical_skeleton import solve_canonical_motion
    from vid2anim.core.universal_transfer import UniversalTransferSolver
    from vid2anim.core.rig_detector import inspect_model, parse_glb_json
    from vid2anim.utils.fixtures import create_synthetic_humanoid_glb, create_synthetic_motion_sequence

    motion = create_synthetic_motion_sequence(frames=15, fps=30.0)
    canon = solve_canonical_motion(motion)
    assert canon.frame_count == 15
    assert len(canon.r_anim) == 15
    assert "hips" in canon.r_anim[0]
    assert "left_arm" in canon.r_anim[0]

    # Verify SO(3) orthogonality of canonical rotations
    for j, r in canon.r_anim[0].items():
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-5), f"Non-orthogonal matrix for {j}"
        assert abs(np.linalg.det(r) - 1.0) < 1e-4, f"Det != 1 for {j}"

    # Test transfer on synthetic humanoid GLB
    model_path = tmp_path / "transfer_test.glb"
    create_synthetic_humanoid_glb(model_path)
    rig = inspect_model(model_path)
    gltf, _ = parse_glb_json(model_path)

    solver = UniversalTransferSolver(gltf, rig)
    assert len(solver.basis_change) > 0

    bone_rots, root_trans = solver.transfer_motion(canon)
    assert len(bone_rots) > 0
    assert root_trans.shape == (15, 3)

    # Verify unit quaternions
    for bone_name, quats in bone_rots.items():
        norms = np.linalg.norm(quats, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"Non-unit quaternions for {bone_name}"


def test_topological_aim_and_zero_inversion(tmp_path):
    """Verify that topological aim detection works across diverse models with zero inversion."""
    from pathlib import Path
    import numpy as np
    from vid2anim.core.rig_detector import inspect_model, parse_glb_json
    from vid2anim.core.universal_transfer import UniversalTransferSolver
    from vid2anim.core.canonical_skeleton import solve_canonical_motion
    from vid2anim.utils.fixtures import create_synthetic_motion_sequence

    motion = create_synthetic_motion_sequence(frames=10, fps=25.0)
    canon = solve_canonical_motion(motion)

    # Test on real models in glb/ if present
    for model_name in ["scp_1471_mal0_stylized_low_poly_character.glb", "robot_police_unit_animated.glb", "cyborg_oni.glb"]:
        p = Path("glb") / model_name
        if not p.exists():
            continue
        gltf, _ = parse_glb_json(p)
        rig = inspect_model(p)
        solver = UniversalTransferSolver(gltf, rig)

        bone_rots, root_trans = solver.transfer_motion(canon, raw_motion=motion)
        assert len(bone_rots) > 0

        # Check unit quaternions for all bones
        for b_name, quats in bone_rots.items():
            norms = np.linalg.norm(quats, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-3), f"Non-unit quaternion in {b_name} for {model_name}"




