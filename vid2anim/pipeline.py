"""High-Level Pipeline Orchestrator for vid2anim.

Manages validation, rig detection, motion extraction, temporal smoothing,
retargeting, and multi-format export with structured error handling and metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.bvh_writer import export_bvh
from .core.calibration import PoseCalibrator, RestPoseType
from .core.glb_writer import GLBAnimationExporter
from .core.motion_extractor import MotionExtractor, MotionSequence
from .core.retargeter import UniversalRetargeter
from .core.rig_detector import RigAnalysis, RigType, inspect_model
from .core.smoothing import TemporalMotionSmoother
from .utils.progress import ProgressReporter


@dataclass
class PipelineConfig:
    """Configuration options for a vid2anim run."""
    input_media: Path
    target_model: Path
    output_glb: Optional[Path] = None
    output_bvh: Optional[Path] = None
    device: str = "auto"
    max_duration_sec: float = 30.0
    smooth: bool = True
    smoothness_factor: float = 1.0
    rest_pose: RestPoseType = RestPoseType.AUTO
    progress_jsonl: Optional[Path] = None


@dataclass
class PipelineResult:
    """Execution metrics and artifact paths returned upon completion."""
    success: bool
    output_glb: Optional[Path] = None
    output_bvh: Optional[Path] = None
    frame_count: int = 0
    duration_sec: float = 0.0
    fps: float = 30.0
    elapsed_time_sec: float = 0.0
    rig_profile: str = "unknown"
    mapped_bones_count: int = 0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class Vid2AnimPipeline:
    """Production orchestrator executing end-to-end video-to-character animation."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.reporter = ProgressReporter(jsonl_path=config.progress_jsonl)

    def execute(self) -> PipelineResult:
        start_time = time.perf_counter()
        warnings: List[str] = []

        try:
            # 1. Validation
            self._validate_inputs()
            out_glb = self.config.output_glb or Path("outputs") / f"{self.config.target_model.stem}_animated.glb"

            # 2. Inspect 3D Model
            self.reporter.report("PIPELINE", 0.05, f"Inspecting 3D model: {self.config.target_model.name}")
            rig_info = inspect_model(self.config.target_model)
            if not rig_info.is_rigged:
                warnings.append("Model is unrigged. Experimental auto-rigging fallback activated.")

            # 3. Extract 3D Motion
            self.reporter.report("PIPELINE", 0.15, f"Extracting motion from {self.config.input_media.name}")
            extractor = MotionExtractor(device=self.config.device, reporter=self.reporter)
            raw_motion = extractor.extract_from_file(
                self.config.input_media,
                max_duration_sec=self.config.max_duration_sec,
            )

            # 4. Outlier Sanitization & Gap Interpolation
            from .core.interpolation import sanitize_and_interpolate_motion
            from .core.scale_normalizer import ScaleNormalizer

            sanitized_motion = sanitize_and_interpolate_motion(raw_motion)

            # Scale trajectory to model coordinate frame
            scale_normalizer = ScaleNormalizer()
            model_scale = scale_normalizer.estimate_model_scale(rig_info)
            scaled_motion = scale_normalizer.scale_trajectory(sanitized_motion, model_scale)

            # 5. Temporal Smoothing (Anti-Jitter Filter)
            motion = scaled_motion
            if self.config.smooth and scaled_motion.frame_count > 2:
                self.reporter.report("PIPELINE", 0.60, "Applying temporal anti-jitter filter...")
                smoother = TemporalMotionSmoother(position_smoothness=self.config.smoothness_factor)
                motion = smoother.smooth_motion_positions(scaled_motion)

            # 6. Universal Retargeting & Rest-Pose Calibration
            pose_desc = f"{rig_info.detected_rest_pose.upper()} ({rig_info.rest_arm_angle_deg:.1f}°)"
            self.reporter.report("PIPELINE", 0.70, f"Retargeting onto {rig_info.rig_type.value} armature [{pose_desc}]...")
            calibrator = PoseCalibrator(
                rest_pose=self.config.rest_pose,
                arm_angle_deg=rig_info.rest_arm_angle_deg,
            )
            retargeter = UniversalRetargeter(reporter=self.reporter)
            animation = retargeter.retarget(motion, rig_info, calibrator=calibrator)

            if self.config.smooth and animation.frame_count > 3:
                smoother = TemporalMotionSmoother()
                animation = smoother.smooth_animation_rotations(animation)

            # 7. Export Skinned Animated GLB
            self.reporter.report("PIPELINE", 0.88, f"Exporting animated GLB to: {out_glb.name}")
            exporter = GLBAnimationExporter(reporter=self.reporter)
            exporter.export_animated_glb(self.config.target_model, animation, rig_info, out_glb)

            # Optional BVH Export
            out_bvh: Optional[Path] = None
            if self.config.output_bvh:
                out_bvh = export_bvh(motion, self.config.output_bvh, reporter=self.reporter)

            elapsed = time.perf_counter() - start_time
            self.reporter.report("PIPELINE", 1.0, f"Finished in {elapsed:.2f}s!")

            return PipelineResult(
                success=True,
                output_glb=out_glb,
                output_bvh=out_bvh,
                frame_count=motion.frame_count,
                duration_sec=motion.duration_sec,
                fps=motion.fps,
                elapsed_time_sec=round(elapsed, 2),
                rig_profile=rig_info.rig_type.value,
                mapped_bones_count=len(animation.bone_rotations),
                warnings=warnings + rig_info.warnings,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.reporter.report("PIPELINE", 1.0, f"Pipeline failed: {str(e)}")
            return PipelineResult(
                success=False,
                elapsed_time_sec=round(elapsed, 2),
                warnings=warnings,
                error_message=str(e),
            )

    def _validate_inputs(self) -> None:
        """Validate input paths and parameter bounds."""
        if not self.config.input_media.exists():
            raise FileNotFoundError(f"Input media does not exist: {self.config.input_media}")
        if not self.config.target_model.exists():
            raise FileNotFoundError(f"Target 3D model does not exist: {self.config.target_model}")
        if self.config.max_duration_sec <= 0.0 or self.config.max_duration_sec > 60.0:
            raise ValueError(f"Max video duration must be between 1 and 60 seconds (got {self.config.max_duration_sec})")
