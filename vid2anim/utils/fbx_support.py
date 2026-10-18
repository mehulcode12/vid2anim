"""Native FBX Model Support & Automatic GLB Conversion.

Provides transparent fallback conversion of .fbx character assets to standardized
binary GLTF (.glb) files so they integrate seamlessly with the animation pipeline.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .progress import ProgressReporter

logger = logging.getLogger("vid2anim.fbx_support")


def convert_fbx_to_glb(
    fbx_path: str | Path,
    output_glb_path: Optional[str | Path] = None,
    reporter: Optional[ProgressReporter] = None,
) -> Path:
    """Convert an FBX character file to binary GLB.
    
    Uses the embedded Three.js / Node.js converter script.
    """
    fbx = Path(fbx_path).resolve()
    if not fbx.exists():
        raise FileNotFoundError(f"FBX model file not found: {fbx}")

    if output_glb_path is not None:
        out_glb = Path(output_glb_path).resolve()
    else:
        out_glb = fbx.with_suffix(".glb")

    out_glb.parent.mkdir(parents=True, exist_ok=True)

    if reporter:
        reporter.report("PIPELINE", 0.06, f"Converting FBX character to GLB: {fbx.name}...")

    # Locate converter script
    script_path = Path(__file__).resolve().parent / "fbx_converter.js"
    if not script_path.exists():
        raise FileNotFoundError(f"FBX converter script not found: {script_path}")

    # Check node executable
    node_exe = shutil.which("node")
    if not node_exe:
        raise RuntimeError(
            "Node.js is required for automatic FBX-to-GLB conversion, but 'node' was not found in PATH."
        )

    # Project root directory containing node_modules
    repo_root = Path(__file__).resolve().parent.parent.parent

    cmd = [node_exe, str(script_path), str(fbx), str(out_glb)]
    logger.info("Executing FBX conversion: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FBX to GLB conversion timed out after 60 seconds on {fbx.name}")

    if proc.returncode != 0 or not out_glb.exists() or out_glb.stat().st_size == 0:
        error_msg = proc.stderr.strip() or proc.stdout.strip() or f"Process exited with code {proc.returncode}"
        raise RuntimeError(f"Failed to convert FBX model '{fbx.name}' to GLB: {error_msg}")

    if reporter:
        reporter.report("PIPELINE", 0.09, f"FBX converted successfully: {out_glb.name}")

    logger.info("FBX converted successfully: %s -> %s (%d bytes)", fbx, out_glb, out_glb.stat().st_size)
    return out_glb


def ensure_glb_model(
    model_path: str | Path,
    reporter: Optional[ProgressReporter] = None,
) -> Path:
    """Ensure that the target model is a binary GLTF (.glb).
    
    If the model is an FBX, converts it automatically in-place or in a temp location.
    """
    p = Path(model_path).resolve()
    if p.suffix.lower() == ".fbx":
        out_glb = p.with_suffix(".converted.glb")
        return convert_fbx_to_glb(p, output_glb_path=out_glb, reporter=reporter)
    return p
