"""Environment and hardware diagnostics for vid2anim."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Any, Dict


def get_system_diagnostics() -> Dict[str, Any]:
    """Inspect environment, hardware (CPU/GPU), FFmpeg, and optional acceleration."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
        "cuda_vram_gb": None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "mediapipe_available": False,
        "torch_available": False,
        "recommended_device": "cpu",
    }

    # Test PyTorch & CUDA availability
    try:
        import torch  # type: ignore

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["cuda_vram_gb"] = round(vram_bytes / (1024**3), 2)
            info["recommended_device"] = "cuda"
    except ImportError:
        info["torch_version"] = None

    # Test MediaPipe (CPU motion extractor backend)
    try:
        import mediapipe  # type: ignore

        info["mediapipe_available"] = True
    except ImportError:
        pass

    return info


def print_doctor_report() -> None:
    """Print user-friendly diagnostics check."""
    info = get_system_diagnostics()

    print("=" * 60)
    print("           vid2anim - System & Hardware Diagnostics")
    print("=" * 60)
    print(f"  OS                  : {info['os']}")
    print(f"  Python Version      : {info['python']}")
    print(f"  CPU Cores           : {info['cpu_count']}")
    
    if info["cuda_available"]:
        print(f"  GPU (CUDA)          : [AVAILABLE] {info['cuda_device_name']} ({info['cuda_vram_gb']} GB VRAM)")
        print("  Default Device      : cuda (High-performance GPU acceleration)")
    else:
        print("  GPU (CUDA)          : [NOT DETECTED / CPU ONLY]")
        print("  Default Device      : cpu (Optimized CPU-friendly pipeline enabled)")
        print("                        (No NVIDIA GPU needed! Video capture will run via CPU)")

    # FFmpeg status
    if info["ffmpeg_available"]:
        print("  FFmpeg              : [OK] Found in system PATH")
    else:
        print("  FFmpeg              : [WARNING] Not found in PATH. Video preview encoding may be limited.")

    # MediaPipe status
    if info["mediapipe_available"]:
        print("  MediaPipe (CPU AI)  : [OK] Ready for real-time CPU pose extraction")
    else:
        print("  MediaPipe           : [MISSING] Install via `pip install mediapipe`")

    print("-" * 60)
    print(f"  Active Mode Target  : {info['recommended_device'].upper()}")
    print("=" * 60)
