"""vid2anim: Animate any 3D model from photo or video.

Supports CPU (default/lightweight) and GPU acceleration.
Universal retargeting for Mixamo, VRM, Ready Player Me, Blender Rigify,
Unreal Engine, Daz3D/Character Creator, and experimental auto-rigging for static meshes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.canonical_skeleton import CANONICAL_JOINTS_ORDER, CanonicalMotion
from .core.motion_extractor import MotionExtractor, MotionSequence
from .core.retargeter import RetargetedAnimation, UniversalRetargeter
from .core.rig_detector import RigAnalysis, RigType, inspect_model
from .pipeline import PipelineConfig, PipelineResult, Vid2AnimPipeline

__all__ = [
    "__version__",
    "CANONICAL_JOINTS_ORDER",
    "CanonicalMotion",
    "MotionExtractor",
    "MotionSequence",
    "PipelineConfig",
    "PipelineResult",
    "RetargetedAnimation",
    "RigAnalysis",
    "RigType",
    "UniversalRetargeter",
    "Vid2AnimPipeline",
    "inspect_model",
]

