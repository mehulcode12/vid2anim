"""Canonical Humanoid Skeleton & Forward Kinematics (FK) Retargeter.

Solves per-frame local rotations hierarchically on a clean, standardized 3D humanoid
armature using exact SO(3) Lie group kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .kinematics import (
    human_to_canonical,
    matrix_to_quat_xyzw,
    normalize,
    orthonormal_frame,
    pick_closest_aim_twist,
    quat_xyzw_to_matrix,
    rotation_aligning,
)
from .motion_extractor import MotionSequence

# Canonical joint names
CANONICAL_JOINTS_ORDER = [
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

# Parent hierarchy
CANONICAL_PARENT: Dict[str, Optional[str]] = {
    "hips": None,
    "spine": "hips",
    "spine1": "spine",
    "chest": "spine1",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_arm": "left_shoulder",
    "left_forearm": "left_arm",
    "left_hand": "left_forearm",
    "right_shoulder": "chest",
    "right_arm": "right_shoulder",
    "right_forearm": "right_arm",
    "right_hand": "right_forearm",
    "left_upleg": "hips",
    "left_leg": "left_upleg",
    "left_foot": "left_leg",
    "left_toe": "left_foot",
    "right_upleg": "hips",
    "right_leg": "right_upleg",
    "right_foot": "right_leg",
    "right_toe": "right_foot",
}

# Rest directions of canonical bones in their local reference frame
CANONICAL_REST_LOCAL_DIR: Dict[str, np.ndarray] = {
    "spine": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "spine1": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "chest": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "neck": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "head": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "left_shoulder": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "left_arm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "left_forearm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "left_hand": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "right_shoulder": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "right_arm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "right_forearm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "right_hand": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "left_upleg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "left_leg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "left_foot": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "left_toe": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "right_upleg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "right_leg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "right_foot": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "right_toe": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}

# Canonical rest world orientation frame F_canonical for each bone:
# columns = [lateral/aim, up, forward]
CANONICAL_REST_WORLD_FRAME: Dict[str, np.ndarray] = {
    "hips": np.eye(3, dtype=np.float64),
    "spine": np.eye(3, dtype=np.float64),
    "spine1": np.eye(3, dtype=np.float64),
    "chest": np.eye(3, dtype=np.float64),
    "neck": np.eye(3, dtype=np.float64),
    "head": np.eye(3, dtype=np.float64),
    "left_shoulder": np.eye(3, dtype=np.float64),
    "left_arm": np.eye(3, dtype=np.float64),
    "left_forearm": np.eye(3, dtype=np.float64),
    "left_hand": np.eye(3, dtype=np.float64),
    "right_shoulder": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
    "right_arm": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
    "right_forearm": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
    "right_hand": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
    "left_upleg": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    "left_leg": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    "left_foot": np.eye(3, dtype=np.float64),
    "left_toe": np.eye(3, dtype=np.float64),
    "right_upleg": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    "right_leg": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    "right_foot": np.eye(3, dtype=np.float64),
    "right_toe": np.eye(3, dtype=np.float64),
}


@dataclass
class CanonicalMotion:
    """Standardized SO(3) Lie algebra motion on the canonical humanoid skeleton."""
    frame_count: int
    fps: float
    duration_sec: float
    timestamps: np.ndarray  # (F,) in seconds
    r_anim: List[Dict[str, np.ndarray]]  # Frame -> (joint -> 3x3 rotation matrix)
    hips_translations: np.ndarray  # (F, 3) in canonical meters
    joint_positions: Optional[Dict[str, np.ndarray]] = None  # (joint -> (F, 3)) in canonical space


def solve_canonical_motion(motion: MotionSequence) -> CanonicalMotion:
    """Solve frame-by-frame FK motion on the canonical humanoid skeleton from joint positions."""
    num_frames = motion.frame_count
    r_anim_frames: List[Dict[str, np.ndarray]] = []
    hips_translations = np.zeros((num_frames, 3), dtype=np.float64)

    prev_r_anim: Dict[str, np.ndarray] = {j: np.eye(3, dtype=np.float64) for j in CANONICAL_JOINTS_ORDER}

    # Pre-extract joint trajectories in canonical space
    def get_pos(j_name: str, f: int) -> np.ndarray:
        if j_name in motion.joint_positions:
            arr = motion.joint_positions[j_name]
            if len(arr) > f:
                raw = arr[f]
            elif len(arr) > 0:
                raw = arr[-1]
            else:
                return np.zeros(3, dtype=np.float64)
            return human_to_canonical(raw)
        return np.zeros(3, dtype=np.float64)

    for f in range(num_frames):
        r_anim: Dict[str, np.ndarray] = {}
        r_world: Dict[str, np.ndarray] = {}

        # 1. Hips Root Orientation & Translation
        p_hips = get_pos("hips", f)
        hips_translations[f] = p_hips

        p_neck = get_pos("neck", f)
        p_l_hip = get_pos("left_upleg", f)
        p_r_hip = get_pos("right_upleg", f)

        v_up = p_neck - p_hips
        v_lat = p_l_hip - p_r_hip
        r_hips = orthonormal_frame(v_lat, v_up)

        r_anim["hips"] = r_hips
        r_world["hips"] = r_hips

        # 2. Spine, Spine1 & Chest (Torso Chain)
        p_spine = get_pos("spine", f)
        p_chest = get_pos("chest", f)
        p_spine1 = get_pos("spine1", f) if "spine1" in motion.joint_positions else (p_spine + p_chest) * 0.5

        # Spine aims along hips -> spine1
        v_spine = p_spine1 - p_spine
        d_spine_in_anim = r_world["hips"].T @ normalize(v_spine, np.array([0.0, 1.0, 0.0]))
        r_spine = rotation_aligning(CANONICAL_REST_LOCAL_DIR["spine"], d_spine_in_anim)
        r_anim["spine"] = r_spine
        r_world["spine"] = r_world["hips"] @ r_spine

        # Spine1 aims along spine1 -> chest
        v_spine1 = p_chest - p_spine1
        d_spine1_in_anim = r_world["spine"].T @ normalize(v_spine1, np.array([0.0, 1.0, 0.0]))
        r_spine1 = rotation_aligning(CANONICAL_REST_LOCAL_DIR["spine1"], d_spine1_in_anim)
        r_anim["spine1"] = r_spine1
        r_world["spine1"] = r_world["spine"] @ r_spine1

        # Chest: Torso chest frame from collar lateral and spine-to-neck up
        p_l_collar = get_pos("left_collar", f) if "left_collar" in motion.joint_positions else get_pos("left_shoulder", f)
        p_r_collar = get_pos("right_collar", f) if "right_collar" in motion.joint_positions else get_pos("right_shoulder", f)
        chest_lat = p_l_collar - p_r_collar
        chest_up = p_neck - p_spine1
        r_chest_world = orthonormal_frame(chest_lat, chest_up)
        r_chest_local = r_world["spine1"].T @ r_chest_world
        r_anim["chest"] = r_chest_local
        r_world["chest"] = r_chest_world

        # Neck & Head (cervical chain sharing orientation smoothly across neck 55% & head 45%)
        # In canonical skeleton, rest cervical axis extends upward [0, 1, 0] along the thoracic spine.
        # Calibrate for the ~16 deg anatomical cervical lordosis angle between chest and cranial center
        # so an upright human looking straight ahead produces a natural level gaze.
        p_head = get_pos("head", f)
        v_neck = p_head - p_chest
        d_neck_raw = r_world["chest"].T @ normalize(v_neck, np.array([0.0, 1.0, 0.0]))
        if d_neck_raw[2] > 0.05:
            theta = float(np.arctan2(d_neck_raw[2], d_neck_raw[1]))
            theta_corr = theta - float(np.radians(16.0))
            d_neck_in_anim = normalize(np.array([
                d_neck_raw[0],
                np.cos(theta_corr),
                np.sin(theta_corr),
            ], dtype=np.float64), np.array([0.0, 1.0, 0.0]))
        else:
            d_neck_in_anim = d_neck_raw
        r_delta_full = rotation_aligning(CANONICAL_REST_LOCAL_DIR["neck"], d_neck_in_anim)
        q_delta = matrix_to_quat_xyzw(r_delta_full)
        half_angle = float(np.arccos(np.clip(q_delta[3], -1.0, 1.0)))
        frac_half = half_angle * 0.55
        sin_h = float(np.sin(half_angle))
        u_axis = q_delta[:3] / sin_h if abs(sin_h) > 1e-6 else np.array([0.0, 1.0, 0.0])
        q_frac = np.array([u_axis[0] * np.sin(frac_half), u_axis[1] * np.sin(frac_half), u_axis[2] * np.sin(frac_half), np.cos(frac_half)], dtype=np.float64)
        r_neck = quat_xyzw_to_matrix(q_frac)
        r_anim["neck"] = r_neck
        r_world["neck"] = r_world["chest"] @ r_neck

        d_head_in_anim = r_world["neck"].T @ (r_world["chest"] @ d_neck_in_anim)
        r_head = rotation_aligning(CANONICAL_REST_LOCAL_DIR["head"], d_head_in_anim)
        r_anim["head"] = r_head
        r_world["head"] = r_world["neck"] @ r_head

        # 3. Left Arm Chain (chest -> left_collar -> left_shoulder -> left_arm -> left_forearm -> left_hand)
        p_l_sh = get_pos("left_shoulder", f)
        p_l_elbow = get_pos("left_forearm", f)
        p_l_wrist = get_pos("left_hand", f)
        p_l_fingers = get_pos("left_fingers", f) if "left_fingers" in motion.joint_positions else (p_l_wrist + (p_l_wrist - p_l_elbow) * 0.35)

        # Shoulder (collar / clavicle bone)
        v_l_sh = p_l_sh - p_l_collar
        d_l_sh_in_anim = r_world["chest"].T @ normalize(v_l_sh, np.array([1.0, 0.0, 0.0]))
        r_l_sh = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_shoulder"], d_l_sh_in_anim)
        r_anim["left_shoulder"] = r_l_sh
        r_world["left_shoulder"] = r_world["chest"] @ r_l_sh

        # Upper arm (shoulder ball joint to elbow)
        v_l_arm = p_l_elbow - p_l_sh
        d_l_arm_in_anim = r_world["left_shoulder"].T @ normalize(v_l_arm, np.array([1.0, 0.0, 0.0]))
        r_l_arm = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_arm"], d_l_arm_in_anim)
        r_l_arm, _ = pick_closest_aim_twist(r_l_arm, prev_r_anim["left_arm"], CANONICAL_REST_LOCAL_DIR["left_arm"])
        r_anim["left_arm"] = r_l_arm
        r_world["left_arm"] = r_world["left_shoulder"] @ r_l_arm

        # Forearm (elbow to wrist)
        v_l_fa = p_l_wrist - p_l_elbow
        d_l_fa_in_anim = r_world["left_arm"].T @ normalize(v_l_fa, np.array([1.0, 0.0, 0.0]))
        r_l_fa = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_forearm"], d_l_fa_in_anim)
        r_l_fa, _ = pick_closest_aim_twist(r_l_fa, prev_r_anim["left_forearm"], CANONICAL_REST_LOCAL_DIR["left_forearm"])
        r_anim["left_forearm"] = r_l_fa
        r_world["left_forearm"] = r_world["left_arm"] @ r_l_fa

        # Hand (wrist to fingers)
        v_l_hand = p_l_fingers - p_l_wrist
        d_l_hand_in_anim = r_world["left_forearm"].T @ normalize(v_l_hand, np.array([1.0, 0.0, 0.0]))
        r_l_hand = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_hand"], d_l_hand_in_anim)
        r_anim["left_hand"] = r_l_hand
        r_world["left_hand"] = r_world["left_forearm"] @ r_l_hand

        # 4. Right Arm Chain (chest -> right_collar -> right_shoulder -> right_arm -> right_forearm -> right_hand)
        p_r_sh = get_pos("right_shoulder", f)
        p_r_elbow = get_pos("right_forearm", f)
        p_r_wrist = get_pos("right_hand", f)
        p_r_fingers = get_pos("right_fingers", f) if "right_fingers" in motion.joint_positions else (p_r_wrist + (p_r_wrist - p_r_elbow) * 0.35)

        # Shoulder (collar / clavicle bone)
        v_r_sh = p_r_sh - p_r_collar
        d_r_sh_in_anim = r_world["chest"].T @ normalize(v_r_sh, np.array([-1.0, 0.0, 0.0]))
        r_r_sh = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_shoulder"], d_r_sh_in_anim)
        r_anim["right_shoulder"] = r_r_sh
        r_world["right_shoulder"] = r_world["chest"] @ r_r_sh

        # Upper arm (shoulder ball joint to elbow)
        v_r_arm = p_r_elbow - p_r_sh
        d_r_arm_in_anim = r_world["right_shoulder"].T @ normalize(v_r_arm, np.array([-1.0, 0.0, 0.0]))
        r_r_arm = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_arm"], d_r_arm_in_anim)
        r_r_arm, _ = pick_closest_aim_twist(r_r_arm, prev_r_anim["right_arm"], CANONICAL_REST_LOCAL_DIR["right_arm"])
        r_anim["right_arm"] = r_r_arm
        r_world["right_arm"] = r_world["right_shoulder"] @ r_r_arm

        # Forearm (elbow to wrist)
        v_r_fa = p_r_wrist - p_r_elbow
        d_r_fa_in_anim = r_world["right_arm"].T @ normalize(v_r_fa, np.array([-1.0, 0.0, 0.0]))
        r_r_fa = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_forearm"], d_r_fa_in_anim)
        r_r_fa, _ = pick_closest_aim_twist(r_r_fa, prev_r_anim["right_forearm"], CANONICAL_REST_LOCAL_DIR["right_forearm"])
        r_anim["right_forearm"] = r_r_fa
        r_world["right_forearm"] = r_world["right_arm"] @ r_r_fa

        # Hand (wrist to fingers)
        v_r_hand = p_r_fingers - p_r_wrist
        d_r_hand_in_anim = r_world["right_forearm"].T @ normalize(v_r_hand, np.array([-1.0, 0.0, 0.0]))
        r_r_hand = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_hand"], d_r_hand_in_anim)
        r_anim["right_hand"] = r_r_hand
        r_world["right_hand"] = r_world["right_forearm"] @ r_r_hand

        # 5. Left Leg Chain (hips -> left_upleg -> left_leg -> left_foot)
        p_l_knee = get_pos("left_leg", f)
        p_l_ankle = get_pos("left_foot", f)

        v_l_thigh = p_l_knee - p_l_hip
        d_l_thigh_in_anim = r_world["hips"].T @ normalize(v_l_thigh, np.array([0.0, -1.0, 0.0]))
        r_l_thigh = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_upleg"], d_l_thigh_in_anim)
        r_l_thigh, _ = pick_closest_aim_twist(r_l_thigh, prev_r_anim["left_upleg"], CANONICAL_REST_LOCAL_DIR["left_upleg"])
        r_anim["left_upleg"] = r_l_thigh
        r_world["left_upleg"] = r_world["hips"] @ r_l_thigh

        v_l_shin = p_l_ankle - p_l_knee
        d_l_shin_in_anim = r_world["left_upleg"].T @ normalize(v_l_shin, np.array([0.0, -1.0, 0.0]))
        r_l_shin = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_leg"], d_l_shin_in_anim)
        r_l_shin, _ = pick_closest_aim_twist(r_l_shin, prev_r_anim["left_leg"], CANONICAL_REST_LOCAL_DIR["left_leg"])
        r_anim["left_leg"] = r_l_shin
        r_world["left_leg"] = r_world["left_upleg"] @ r_l_shin

        p_l_toe = get_pos("left_toe", f) if "left_toe" in motion.joint_positions else p_l_ankle + np.array([0.0, -0.05, 0.15])
        v_l_foot = p_l_toe - p_l_ankle
        d_l_foot_in_anim = r_world["left_leg"].T @ normalize(v_l_foot, np.array([0.0, 0.0, 1.0]))
        r_l_foot = rotation_aligning(CANONICAL_REST_LOCAL_DIR["left_foot"], d_l_foot_in_anim)
        r_anim["left_foot"] = r_l_foot
        r_world["left_foot"] = r_world["left_leg"] @ r_l_foot

        dy_l_toe = float(p_l_ankle[1] - p_l_toe[1])
        pitch_l = float(np.arcsin(np.clip(dy_l_toe / 0.12, 0.0, 0.70))) if dy_l_toe > 0.02 else 0.0
        c_l, s_l = np.cos(pitch_l), np.sin(pitch_l)
        r_toe_l = np.array([[1.0, 0.0, 0.0], [0.0, c_l, -s_l], [0.0, s_l, c_l]], dtype=np.float64)
        r_anim["left_toe"] = r_toe_l
        r_world["left_toe"] = r_world["left_foot"] @ r_toe_l

        # 6. Right Leg Chain (hips -> right_upleg -> right_leg -> right_foot -> right_toe)
        p_r_knee = get_pos("right_leg", f)
        p_r_ankle = get_pos("right_foot", f)

        v_r_thigh = p_r_knee - p_r_hip
        d_r_thigh_in_anim = r_world["hips"].T @ normalize(v_r_thigh, np.array([0.0, -1.0, 0.0]))
        r_r_thigh = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_upleg"], d_r_thigh_in_anim)
        r_r_thigh, _ = pick_closest_aim_twist(r_r_thigh, prev_r_anim["right_upleg"], CANONICAL_REST_LOCAL_DIR["right_upleg"])
        r_anim["right_upleg"] = r_r_thigh
        r_world["right_upleg"] = r_world["hips"] @ r_r_thigh

        v_r_shin = p_r_ankle - p_r_knee
        d_r_shin_in_anim = r_world["right_upleg"].T @ normalize(v_r_shin, np.array([0.0, -1.0, 0.0]))
        r_r_shin = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_leg"], d_r_shin_in_anim)
        r_r_shin, _ = pick_closest_aim_twist(r_r_shin, prev_r_anim["right_leg"], CANONICAL_REST_LOCAL_DIR["right_leg"])
        r_anim["right_leg"] = r_r_shin
        r_world["right_leg"] = r_world["right_upleg"] @ r_r_shin

        p_r_toe = get_pos("right_toe", f) if "right_toe" in motion.joint_positions else p_r_ankle + np.array([0.0, -0.05, 0.15])
        v_r_foot = p_r_toe - p_r_ankle
        d_r_foot_in_anim = r_world["right_leg"].T @ normalize(v_r_foot, np.array([0.0, 0.0, 1.0]))
        r_r_foot = rotation_aligning(CANONICAL_REST_LOCAL_DIR["right_foot"], d_r_foot_in_anim)
        r_anim["right_foot"] = r_r_foot
        r_world["right_foot"] = r_world["right_leg"] @ r_r_foot

        dy_r_toe = float(p_r_ankle[1] - p_r_toe[1])
        pitch_r = float(np.arcsin(np.clip(dy_r_toe / 0.12, 0.0, 0.70))) if dy_r_toe > 0.02 else 0.0
        c_r, s_r = np.cos(pitch_r), np.sin(pitch_r)
        r_toe_r = np.array([[1.0, 0.0, 0.0], [0.0, c_r, -s_r], [0.0, s_r, c_r]], dtype=np.float64)
        r_anim["right_toe"] = r_toe_r
        r_world["right_toe"] = r_world["right_foot"] @ r_toe_r

        # Commit frame
        r_anim_frames.append(r_anim)
        for j in CANONICAL_JOINTS_ORDER:
            prev_r_anim[j] = r_anim[j]

    # Canonical joint positions
    canon_joint_positions = {
        j: np.array([get_pos(j, f) for f in range(num_frames)], dtype=np.float64)
        for j in CANONICAL_JOINTS_ORDER
        if j in motion.joint_positions
    }

    return CanonicalMotion(
        frame_count=num_frames,
        fps=motion.fps,
        duration_sec=motion.duration_sec,
        timestamps=motion.timestamps,
        r_anim=r_anim_frames,
        hips_translations=hips_translations,
        joint_positions=canon_joint_positions,
    )

