"""Command Line Interface for vid2anim.

Commands:
  vid2anim doctor     Check hardware, CPU, CUDA GPU, and dependencies.
  vid2anim inspect    Inspect a 3D model and detect its skeleton/rig profile.
  vid2anim extract    Extract 3D motion from a video or photo to cached .npz.
  vid2anim retarget   Retarget cached motion onto a 3D model.
  vid2anim run        End-to-end: input media + 3D model -> animated .glb!
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.glb_writer import GLBAnimationExporter
from .core.motion_extractor import MotionExtractor, MotionSequence
from .core.retargeter import UniversalRetargeter
from .core.rig_detector import RigType, inspect_model
from .utils.diagnostics import print_doctor_report
from .utils.progress import ProgressReporter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vid2anim",
        description="Animate any 3D model from photo or video with AI motion capture and universal retargeting.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: doctor
    subparsers.add_parser("doctor", help="Check hardware, CPU, CUDA GPU, FFmpeg, and dependencies.")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a 3D model's armature, format, and bone mapping.")
    inspect_parser.add_argument("model", help="Path to 3D asset (.glb, .gltf, .fbx, .obj)")

    # Command: extract
    extract_parser = subparsers.add_parser("extract", help="Extract 3D motion from video or photo to cached .npz.")
    extract_parser.add_argument("--input", "-i", required=True, help="Path to input photo or video (up to 30s)")
    extract_parser.add_argument("--output", "-o", default="outputs/motion_cache.npz", help="Output path for cached .npz")
    extract_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device (default: auto)")
    extract_parser.add_argument("--max-duration", type=float, default=30.0, help="Max video duration in seconds (default: 30.0)")

    # Command: retarget
    retarget_parser = subparsers.add_parser("retarget", help="Retarget cached motion onto a 3D model.")
    retarget_parser.add_argument("--motion", "-m", required=True, help="Path to cached motion .npz")
    retarget_parser.add_argument("--model", required=True, help="Path to 3D character model (.glb)")
    retarget_parser.add_argument("--output", "-o", default="outputs/animated_character.glb", help="Output path for animated .glb")

    # Command: serve (Web UI)
    serve_parser = subparsers.add_parser("serve", help="Launch local Web UI & API server with 3D viewer.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # Command: run (End-to-End)
    run_parser = subparsers.add_parser("run", help="Full pipeline: media + 3D model -> animated .glb")
    run_parser.add_argument("--input", "-i", required=True, help="Input video or photo (.mp4, .jpg, etc.)")
    run_parser.add_argument("--model", "-m", required=True, help="Target 3D character asset (.glb, .fbx, .obj)")
    run_parser.add_argument("--output", "-o", default=None, help="Output animated .glb path (default: outputs/<name>_animated.glb)")
    run_parser.add_argument("--bvh", default=None, help="Optional output path to export standard .bvh mocap")
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device (default: auto)")
    run_parser.add_argument("--max-duration", type=float, default=30.0, help="Max video duration in seconds (default: 30.0)")
    run_parser.add_argument("--no-smooth", action="store_true", help="Disable temporal anti-jitter filter")
    run_parser.add_argument("--rest-pose", choices=["t_pose", "a_pose_30", "a_pose_45"], default="t_pose", help="Target model rest pose alignment (default: t_pose)")
    run_parser.add_argument("--progress-jsonl", default=None, help="Path to write streaming JSONL progress events (for Web UI / extensions)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "doctor":
        print_doctor_report()
        sys.exit(0)

    elif args.command == "inspect":
        analysis = inspect_model(args.model)
        print("=" * 60)
        print(f"3D Model Inspection: {analysis.file_path.name}")
        print("=" * 60)
        print(f"  Format        : {analysis.format}")
        print(f"  Is Rigged     : {'YES' if analysis.is_rigged else 'NO (Static Mesh)'}")
        print(f"  Rig Profile   : {analysis.rig_type.value.upper()}")
        print(f"  Total Bones   : {len(analysis.all_bones)}")
        print(f"  Mapped Joints : {len(analysis.bone_mapping)}/19 canonical humanoid joints")
        if analysis.bone_mapping:
            print("\n  Sample Mappings:")
            for c_joint, target in list(analysis.bone_mapping.items())[:6]:
                print(f"    - {c_joint:15s} -> {target}")
        if analysis.warnings:
            print("\n  Warnings:")
            for w in analysis.warnings:
                print(f"    [!] {w}")
        print("=" * 60)
        sys.exit(0)

    elif args.command == "serve":
        from .server import start_server
        start_server(host=args.host, port=args.port)
        sys.exit(0)

    elif args.command == "extract":
        reporter = ProgressReporter()
        extractor = MotionExtractor(device=args.device, reporter=reporter)
        motion = extractor.extract_from_file(args.input, max_duration_sec=args.max_duration)
        motion.save_cache(args.output)
        print(f"\n[DONE] Motion saved to {args.output} ({motion.frame_count} frames, {motion.fps:.1f} FPS)")
        sys.exit(0)

    elif args.command == "retarget":
        reporter = ProgressReporter()
        motion = MotionSequence.load_cache(args.motion)
        rig_info = inspect_model(args.model)
        retargeter = UniversalRetargeter(reporter=reporter)
        anim = retargeter.retarget(motion, rig_info)
        exporter = GLBAnimationExporter(reporter=reporter)
        exporter.export_animated_glb(args.model, anim, rig_info, args.output)
        print(f"\n[DONE] Animated model exported to {args.output}")
        sys.exit(0)

    elif args.command == "run":
        from .core.calibration import RestPoseType
        from .pipeline import PipelineConfig, Vid2AnimPipeline

        config = PipelineConfig(
            input_media=Path(args.input),
            target_model=Path(args.model),
            output_glb=Path(args.output) if args.output else None,
            output_bvh=Path(args.bvh) if args.bvh else None,
            device=args.device,
            max_duration_sec=args.max_duration,
            smooth=not args.no_smooth,
            rest_pose=RestPoseType(args.rest_pose),
            progress_jsonl=Path(args.progress_jsonl) if args.progress_jsonl else None,
        )

        pipeline = Vid2AnimPipeline(config)
        result = pipeline.execute()

        if result.success:
            print("=" * 60)
            print(f"SUCCESS: Animated character exported to:\n  {result.output_glb}")
            if result.output_bvh:
                print(f"MoCap BVH file saved to:\n  {result.output_bvh}")
            print(f"Processed {result.frame_count} frames ({result.duration_sec:.2f}s) in {result.elapsed_time_sec}s.")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print(f"ERROR: Execution failed: {result.error_message}")
            print("=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
