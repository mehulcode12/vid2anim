"""Standard Biovision Hierarchy (.bvh) Motion Capture Exporter.

Exports universal MoCap data compatible with Blender, Maya, Unreal Engine,
Unity, and any 3D animation software.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from ..utils.math_3d import normalize
from ..utils.progress import ProgressReporter
from .motion_extractor import MotionSequence


def export_bvh(
    motion: MotionSequence,
    output_path: str | Path,
    reporter: Optional[ProgressReporter] = None,
) -> Path:
    """Write extracted motion sequence into standard BVH format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = reporter or ProgressReporter()
    rep.report("EXPORT_BVH", 0.1, f"Generating BVH motion capture: {out.name}")

    frame_time = 1.0 / motion.fps if motion.fps > 0 else 0.033333

    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "\tOFFSET 0.00 0.00 0.00",
        "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        "\tJOINT Spine",
        "\t{",
        "\t\tOFFSET 0.00 15.00 0.00",
        "\t\tCHANNELS 3 Zrotation Xrotation Yrotation",
        "\t\tEnd Site",
        "\t\t{",
        "\t\t\tOFFSET 0.00 20.00 0.00",
        "\t\t}",
        "\t}",
        "}",
        "MOTION",
        f"Frames: {motion.frame_count}",
        f"Frame Time: {frame_time:.6f}",
    ]

    # Motion frames: write root translation + 0 rotation
    for f in range(motion.frame_count):
        pos = motion.root_translation[f] * 100.0  # meters to cm
        # X Y Z position, Z X Y rotation for hips, Z X Y for spine
        lines.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rep.report("EXPORT_BVH", 1.0, f"Saved BVH to {out.name}")
    return out
