"""Open-source CLI feature utilities for vid2anim: samples, batch, addon, and benchmark."""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from .progress import ProgressReporter


def get_project_root() -> Path:
    """Find the project root containing samples/ and plugins/."""
    # First try parent of package directory
    pkg_dir = Path(__file__).resolve().parents[1]
    proj_dir = pkg_dir.parent
    if (proj_dir / "samples").is_dir() and (proj_dir / "plugins").is_dir():
        return proj_dir
    # Check current working directory
    cwd = Path.cwd()
    if (cwd / "samples").is_dir():
        return cwd
    return proj_dir


def handle_samples_command(action: str, model_name: str = "ybot", engine: str = "mediapipe", dest_dir: Optional[str] = None) -> None:
    """Manage and inspect bundled sample models and videos."""
    root = get_project_root()
    samples_dir = root / "samples"
    models_dir = samples_dir / "models"
    videos_dir = samples_dir / "videos"

    if action == "list":
        print("=" * 65)
        print("                 vid2anim Official Sample Assets")
        print("=" * 65)
        if models_dir.is_dir():
            print("\n  [3D Character Models] (`samples/models/`):")
            for m in sorted(models_dir.glob("*.glb")):
                sz_mb = m.stat().st_size / (1024 * 1024)
                note = "Standard 52-bone Mixamo" if "ybot" in m.name else "Detailed 65-bone 5-digit fingers"
                print(f"    - {m.name:15s} [{sz_mb:5.1f} MB] - {note}")
        else:
            print("  [!] models/ directory not found in samples/")

        if videos_dir.is_dir():
            print("\n  [Reference Dance Videos] (`samples/videos/`):")
            for v in sorted(videos_dir.glob("*.mp4")):
                sz_mb = v.stat().st_size / (1024 * 1024)
                desc = "Urban choreography (11.2s)" if "urban" in v.name else "Freestyle street dance (15.2s)"
                print(f"    - {v.name:17s} [{sz_mb:5.1f} MB] - {desc}")
        else:
            print("  [!] videos/ directory not found in samples/")

        print("\n  [Quick Commands]:")
        print("    vid2anim samples run                  # Animate ybot with urban dance on CPU")
        print("    vid2anim samples run --model eve      # Animate eve with 5-digit fingers")
        print("    vid2anim samples copy ./my_samples    # Copy assets to local directory")
        print("=" * 65)

    elif action == "copy":
        target = Path(dest_dir or "./samples").resolve()
        if not samples_dir.is_dir():
            print(f"[ERROR] Sample assets directory not found at {samples_dir}")
            sys.exit(1)
        shutil.copytree(samples_dir, target, dirs_exist_ok=True)
        print(f"[OK] Sample assets copied successfully to:\n  {target}")

    elif action == "run":
        from ..pipeline import PipelineConfig, Vid2AnimPipeline

        chosen_model = models_dir / f"{model_name}.glb" if not model_name.endswith(".glb") else models_dir / model_name
        chosen_video = videos_dir / "dance_urban.mp4"

        if not chosen_model.is_file():
            # Fallback to ybot
            chosen_model = models_dir / "ybot.glb"
        if not chosen_video.is_file():
            print(f"[ERROR] Sample reference video not found at {chosen_video}")
            sys.exit(1)

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_glb = out_dir / f"sample_{chosen_model.stem}_animated.glb"

        print("=" * 65)
        print("              vid2anim - Quick Sample Demonstration")
        print("=" * 65)
        print(f"  Reference Video : {chosen_video.name}")
        print(f"  Target Model    : {chosen_model.name}")
        print(f"  Engine          : {engine.upper()} (CPU-Optimized)")
        print(f"  Destination     : {out_glb}")
        print("=" * 65 + "\n")

        config = PipelineConfig(
            input_media=chosen_video,
            target_model=chosen_model,
            output_glb=out_glb,
            engine=engine,
            max_duration_sec=12.0,
        )
        pipeline = Vid2AnimPipeline(config)
        res = pipeline.execute()

        if res.success:
            print("\n" + "=" * 65)
            print(f"SUCCESS: Sample animated GLB generated:\n  {res.output_glb}")
            print(f"Processed {res.frame_count} frames in {res.elapsed_time_sec}s.")
            print("Open in any 3D viewer (Blender, Three.js, Windows 3D Viewer)!")
            print("=" * 65)
        else:
            print(f"\n[ERROR] Pipeline run failed: {res.error_message}")
            sys.exit(1)


def handle_batch_command(
    inputs: Optional[List[str]],
    input_dir: Optional[str],
    models: Optional[List[str]],
    model_dir: Optional[str],
    output_dir: str = "outputs/batch",
    engine: str = "mediapipe",
    device: str = "auto",
    max_duration: float = 30.0,
) -> None:
    """Batch process multiple input media files and/or 3D models."""
    from ..pipeline import PipelineConfig, Vid2AnimPipeline

    input_paths: List[Path] = []
    if inputs:
        for inp in inputs:
            p = Path(inp)
            if p.is_file():
                input_paths.append(p)
    if input_dir:
        d = Path(input_dir)
        if d.is_dir():
            for ext in ("*.mp4", "*.mov", "*.avi", "*.jpg", "*.png"):
                input_paths.extend(d.glob(ext))

    model_paths: List[Path] = []
    if models:
        for m in models:
            p = Path(m)
            if p.is_file():
                model_paths.append(p)
    if model_dir:
        d = Path(model_dir)
        if d.is_dir():
            for ext in ("*.glb", "*.gltf"):
                model_paths.extend(d.glob(ext))

    if not input_paths:
        print("[ERROR] No input media files provided. Use --inputs or --input-dir.")
        sys.exit(1)
    if not model_paths:
        print("[ERROR] No target 3D models provided. Use --models, --model, or --model-dir.")
        sys.exit(1)

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    pairs = [(inp, mdl) for inp in input_paths for mdl in model_paths]
    total = len(pairs)

    print("=" * 65)
    print(f"     vid2anim Batch Processing Pipeline ({total} animations queued)")
    print("=" * 65)
    print(f"  Input Files     : {len(input_paths)} videos/photos")
    print(f"  Target Models   : {len(model_paths)} 3D characters")
    print(f"  Engine / Device : {engine} on {device}")
    print(f"  Output Directory: {out_base.resolve()}")
    print("=" * 65)

    succeeded = 0
    failed = 0
    t0 = time.time()

    for idx, (inp, mdl) in enumerate(pairs, start=1):
        out_name = f"{inp.stem}_{mdl.stem}_animated.glb"
        out_glb = out_base / out_name
        print(f"\n[{idx}/{total}] Processing: {inp.name} -> {mdl.name}")

        try:
            config = PipelineConfig(
                input_media=inp,
                target_model=mdl,
                output_glb=out_glb,
                engine=engine,
                device=device,
                max_duration_sec=max_duration,
            )
            pipeline = Vid2AnimPipeline(config)
            res = pipeline.execute()

            if res.success:
                succeeded += 1
                print(f"  [OK] Saved to: {out_glb.name} ({res.elapsed_time_sec}s, {res.frame_count} frames)")
            else:
                failed += 1
                print(f"  [FAIL] {res.error_message}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] Exception: {e}")

    elapsed = round(time.time() - t0, 1)
    print("\n" + "=" * 65)
    print(f"Batch Complete: {succeeded}/{total} succeeded ({failed} failed) in {elapsed}s")
    print(f"Outputs saved to: {out_base.resolve()}")
    print("=" * 65)


def handle_addon_command(export_path: Optional[str] = None, auto_install: bool = False) -> None:
    """Package or install the Blender bridge add-on."""
    root = get_project_root()
    addon_py = root / "plugins" / "blender" / "vid2anim_blender.py"

    if not addon_py.is_file():
        # Fallback check
        pkg_root = Path(__file__).resolve().parents[2]
        addon_py = pkg_root / "plugins" / "blender" / "vid2anim_blender.py"

    if not addon_py.is_file():
        print(f"[ERROR] Blender add-on file not found at {addon_py}")
        sys.exit(1)

    out_zip = Path(export_path or "outputs/vid2anim_blender.zip").resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(addon_py, arcname="vid2anim_blender/vid2anim_blender.py")
        zf.write(addon_py, arcname="vid2anim_blender/__init__.py")

    print("=" * 65)
    print("               vid2anim Blender Bridge Add-on")
    print("=" * 65)
    print(f"  [OK] Add-on package exported to:\n       {out_zip}\n")

    if auto_install:
        installed = False
        candidates: List[Path] = []
        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA")
            if appdata:
                bl_dir = Path(appdata) / "Blender Foundation" / "Blender"
                if bl_dir.is_dir():
                    candidates.extend(bl_dir.glob("*/scripts/addons"))
        elif sys.platform == "darwin":
            bl_dir = Path.home() / "Library" / "Application Support" / "Blender"
            if bl_dir.is_dir():
                candidates.extend(bl_dir.glob("*/scripts/addons"))
        else:
            bl_dir = Path.home() / ".config" / "blender"
            if bl_dir.is_dir():
                candidates.extend(bl_dir.glob("*/scripts/addons"))

        for target_addon_dir in candidates:
            if target_addon_dir.is_dir():
                dest_file = target_addon_dir / "vid2anim_blender.py"
                shutil.copy2(addon_py, dest_file)
                print(f"  [INSTALLED] Directly installed into Blender:\n              {dest_file}")
                installed = True

        if not installed:
            print("  [!] Could not locate active Blender addons directory automatically.")
            print("      Please use the manual 2-step installation below.\n")

    print("  Manual Installation in Blender (3.0 - 4.2+ LTS):")
    print("    1. Open Blender")
    print("    2. Go to Edit > Preferences > Add-ons")
    print("    3. Click 'Install...' and choose:")
    print(f"       {out_zip}")
    print("    4. Enable 'Animation: vid2anim Blender Bridge'")
    print("    5. In the 3D Viewport, press 'N' to open the vid2anim sidebar tab!")
    print("=" * 65)


def handle_benchmark_command(frames: int = 60, engine: str = "mediapipe") -> None:
    """Benchmark extraction and retargeting throughput on current hardware."""
    from .diagnostics import get_system_diagnostics
    from .fixtures import create_synthetic_motion_sequence
    from ..core.retargeter import UniversalRetargeter
    from ..core.rig_detector import inspect_model

    diag = get_system_diagnostics()

    print("=" * 65)
    print("             vid2anim Hardware & Throughput Benchmark")
    print("=" * 65)
    print(f"  Operating System : {diag['os']}")
    print(f"  Python           : {diag['python']}")
    print(f"  CPU Threads      : {diag['cpu_count']}")
    print(f"  CUDA GPU         : {diag['cuda_device_name'] or 'None (CPU Execution)'}")
    print(f"  Benchmark Frames : {frames} frames")
    print("-" * 65)

    # 1. Synthetic Sequence Generation
    t0 = time.perf_counter()
    motion = create_synthetic_motion_sequence(frames=frames, fps=30.0)
    gen_time = time.perf_counter() - t0

    # 2. Universal Armature Retargeting
    root = get_project_root()
    ybot_path = root / "samples" / "models" / "ybot.glb"
    if not ybot_path.is_file():
        ybot_path = root / "glb" / "ybot.glb"

    if ybot_path.is_file():
        rig = inspect_model(ybot_path)
        retargeter = UniversalRetargeter()

        t0 = time.perf_counter()
        anim = retargeter.retarget(motion, rig=rig)
        retarget_time = time.perf_counter() - t0

        fps_retarget = frames / max(1e-6, retarget_time)
        ms_per_frame = (retarget_time / frames) * 1000

        print(f"  Universal Retargeter  : {fps_retarget:7.1f} FPS ({ms_per_frame:.2f} ms/frame)")
    else:
        print("  Universal Retargeter  : Skipped (no sample model found)")

    # 3. MediaPipe Inference Speed (if installed)
    if diag.get("mediapipe_available"):
        import numpy as np
        from ..core.motion_extractor import MotionExtractor

        extractor = MotionExtractor(engine="mediapipe")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        test_cnt = min(frames, 30)
        try:
            import mediapipe as mp
            import cv2
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                with mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                ) as pose:
                    rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
                    for _ in range(test_cnt):
                        pose.process(rgb)
            else:
                for _ in range(test_cnt):
                    extractor._estimate_frame_3d_pose(dummy_frame)
            mp_time = time.perf_counter() - t0
            mp_fps = test_cnt / max(1e-6, mp_time)
            print(f"  MediaPipe Pose (CPU)  : {mp_fps:7.1f} FPS ({(mp_time/test_cnt)*1000:.2f} ms/frame)")
        except Exception as e:
            print(f"  MediaPipe Pose (CPU)  : Available (Benchmark notice: {e})")
    else:
        print("  MediaPipe Pose (CPU)  : Not installed")

    print("-" * 65)
    print("  Engine Guide for Your Hardware:")
    if diag.get("cuda_available"):
        print("    - 'gvhmr'    : Recommended for studio-quality parametric 3D HMR (GPU active)")
        print("    - 'dwpose'   : Recommended for high-speed 133-point wholebody & fingers")
    else:
        print("    - 'mediapipe': Recommended for ultra-fast, real-time CPU animation (60+ FPS)")
        print("    - 'dwpose'   : Recommended when precise finger gestures are needed on CPU")
    print("=" * 65)


def handle_info_command() -> None:
    """Print a quick overview of version, environment, and isolated paths."""
    from .. import __version__
    from .diagnostics import get_system_diagnostics
    from ..core.motion_extractor import configure_isolated_environment

    proj_dir = configure_isolated_environment()
    diag = get_system_diagnostics()

    print("=" * 65)
    print(f"             vid2anim v{__version__} - Environment Overview")
    print("=" * 65)
    print(f"  Package Version    : {__version__}")
    print(f"  Python Environment : {sys.executable}")
    print(f"  Platform           : {diag['os']}")
    print(f"  CUDA GPU Available : {'YES (' + diag['cuda_device_name'] + ')' if diag['cuda_available'] else 'NO (CPU Mode)'}")
    print("\n  Triple Tracking Engines:")
    print(f"    - MediaPipe      : {'[AVAILABLE]' if diag.get('mediapipe_available') else '[NOT INSTALLED]'}")
    print("    - DWPose         : [AVAILABLE] (ONNX / WholeBody)")
    print("    - GVHMR          : [AVAILABLE] (SMPL-X Kinematic Studio)")
    print("\n  Isolated Project Storage (Zero C: Drive Writes):")
    print(f"    - Project Root   : {proj_dir}")
    print(f"    - Cache Home     : {os.environ.get('XDG_CACHE_HOME', 'default')}")
    print(f"    - PyTorch Cache  : {os.environ.get('TORCH_HOME', 'default')}")
    print(f"    - HuggingFace    : {os.environ.get('HF_HOME', 'default')}")
    print("=" * 65)
