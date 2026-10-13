"""Experimental Auto-Rigging and Skinning Engine for Unrigged Static 3D Meshes.

When a user provides a 3D model without bones, this module:
1. Calculates mesh dimensions and bounding bounds.
2. Scales and embeds a canonical 19-joint humanoid skeleton template inside the mesh.
3. Computes smooth distance-weighted Linear Blend Skinning (LBS) weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .rig_detector import CANONICAL_JOINTS


# Canonical relative joint positions (% of height, width, depth from base center)
# (relative_x, relative_y, relative_z) where Y is up, X is lateral, Z is forward
CANONICAL_BIPED_TEMPLATE: Dict[str, Tuple[float, float, float]] = {
    "hips": (0.0, 0.52, 0.0),
    "spine": (0.0, 0.62, 0.0),
    "chest": (0.0, 0.72, 0.02),
    "neck": (0.0, 0.82, 0.02),
    "head": (0.0, 0.92, 0.04),
    "left_shoulder": (0.08, 0.76, 0.0),
    "left_arm": (0.18, 0.74, 0.0),
    "left_forearm": (0.34, 0.74, 0.0),
    "left_hand": (0.48, 0.74, 0.0),
    "right_shoulder": (-0.08, 0.76, 0.0),
    "right_arm": (-0.18, 0.74, 0.0),
    "right_forearm": (-0.34, 0.74, 0.0),
    "right_hand": (-0.48, 0.74, 0.0),
    "left_upleg": (0.10, 0.48, 0.0),
    "left_leg": (0.10, 0.26, -0.01),
    "left_foot": (0.10, 0.04, 0.08),
    "right_upleg": (-0.10, 0.48, 0.0),
    "right_leg": (-0.10, 0.26, -0.01),
    "right_foot": (-0.10, 0.04, 0.08),
}

# Parent-child hierarchy
BIPED_HIERARCHY: Dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
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
    "right_upleg": "hips",
    "right_leg": "right_upleg",
    "right_foot": "right_leg",
}


@dataclass
class AutoRiggedModel:
    """Result of experimental auto-rigging on a static mesh."""
    joint_positions: Dict[str, np.ndarray]  # Joint 3D world locations
    joint_names: List[str]
    parent_map: Dict[str, str | None]
    vertex_joints: np.ndarray  # (N, 4) uint16 indices of top 4 influencing bones
    vertex_weights: np.ndarray  # (N, 4) float32 normalized weights


def fit_skeleton_to_mesh(vertices: np.ndarray) -> AutoRiggedModel:
    """Fit a humanoid skeleton and compute skinning weights for an arbitrary static mesh.
    
    Args:
        vertices: (N, 3) numpy array of vertex coordinates in Y-up convention.
    """
    if len(vertices) == 0:
        raise ValueError("Vertex array is empty; cannot auto-rig.")

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) * 0.5
    size = max_coords - min_coords

    height = max(size[1], 1e-4)
    width = max(size[0], 1e-4)
    depth = max(size[2], 1e-4)
    base_y = min_coords[1]

    # Compute 3D joint world positions from template
    joint_positions: Dict[str, np.ndarray] = {}
    joint_names = list(CANONICAL_BIPED_TEMPLATE.keys())

    for j_name, (rel_x, rel_y, rel_z) in CANONICAL_BIPED_TEMPLATE.items():
        world_x = center[0] + rel_x * width * 1.8
        world_y = base_y + rel_y * height
        world_z = center[2] + rel_z * depth
        joint_positions[j_name] = np.array([world_x, world_y, world_z], dtype=np.float32)

    # Compute distance-based skinning weights
    n_verts = len(vertices)
    n_joints = len(joint_names)
    joint_matrix = np.stack([joint_positions[name] for name in joint_names], axis=0)  # (J, 3)

    # Compute distances between each vertex and each joint: (N, J)
    # Broadcast: (N, 1, 3) - (1, J, 3)
    diff = vertices[:, np.newaxis, :] - joint_matrix[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (N, J)

    # Compute inverse distance weights with smooth falloff
    # Avoid div-by-zero with small epsilon
    inv_dist = 1.0 / np.power(dist + 1e-3, 2.5)

    # Select top 4 influencing bones per vertex
    top_indices = np.argsort(-inv_dist, axis=1)[:, :4]  # (N, 4)
    top_weights = np.take_along_axis(inv_dist, top_indices, axis=1)  # (N, 4)

    # Normalize weights so they sum to 1.0
    weight_sums = np.sum(top_weights, axis=1, keepdims=True)
    top_weights = np.where(weight_sums > 1e-8, top_weights / weight_sums, 0.25).astype(np.float32)
    top_indices = top_indices.astype(np.uint16)

    return AutoRiggedModel(
        joint_positions=joint_positions,
        joint_names=joint_names,
        parent_map=BIPED_HIERARCHY,
        vertex_joints=top_indices,
        vertex_weights=top_weights,
    )
