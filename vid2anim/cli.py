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

    # Command: run (End-to-End)
    run_parser = subparsers.add_parser("run", help="Full pipeline: media + 3D model -> animated .glb")
    run_parser.add_argument("--input", "-i", required=True, help="Input video or photo (.mp4, .jpg, etc.)")
    run_parser.add_argument("--model", "-m", required=True, help="Target 3D character asset (.glb, .fbx, .obj)")
    run_parser.add_argument("--output", "-o", default=None, help="Output animated .glb path (default: outputs/<name>_animated.glb)")
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device (default: auto)")
    run_parser.add_argument("--max-duration", type=float, default=30.0, help="Max video duration in seconds (default: 30.0)")
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
        reporter = ProgressReporter(jsonl_path=args.progress_jsonl)
        input_path = Path(args.input)
        model_path = Path(args.model)

        out_path = Path(args.output) if args.output else Path("outputs") / f"{model_path.stem}_animated.glb"

        print("=" * 60)
        print("               vid2anim - Pipeline Execution")
        print("=" * 60)
        print(f"  Input Media : {input_path}")
        print(f"  Target Model: {model_path}")
        print(f"  Device      : {args.device.upper()}")
        print(f"  Output      : {out_path}")
        print("-" * 60)

        # 1. Rig Inspection
        reporter.report("PIPELINE", 0.05, f"Inspecting 3D model: {model_path.name}")
        rig_info = inspect_model(model_path)
        if not rig_info.is_rigged:
            reporter.report(
                "PIPELINE",
                0.08,
                "[WARNING] Model has no bones. Running Experimental Auto-Rigger...",
            )

        # 2. Motion Extraction
        extractor = MotionExtractor(device=args.device, reporter=reporter)
        motion = extractor.extract_from_file(input_path, max_duration_sec=args.max_duration)

        # 3. Retargeting
        retargeter = UniversalRetargeter(reporter=reporter)
        anim = retargeter.retarget(motion, rig_info)

        # 4. Export
        exporter = GLBAnimationExporter(reporter=reporter)
        exporter.export_animated_glb(model_path, anim, rig_info, out_path)

        reporter.report("PIPELINE", 1.0, f"Complete! Output saved to: {out_path}")
        print("=" * 60)
        print(f"SUCCESS: Your animated 3D character is ready at:\n  {out_path}")
        print("You can open this .glb directly in Blender, Unity, or Windows 3D Viewer!")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
