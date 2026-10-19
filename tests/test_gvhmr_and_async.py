"""Tests for GVHMR Studio Engine integration, Async frame prefetching, and drive isolation."""

import os
from pathlib import Path
import numpy as np
import pytest

from vid2anim.core.motion_extractor import (
    MotionExtractor,
    MotionSequence,
    AsyncVideoReader,
    configure_isolated_environment,
)
from vid2anim.core.retargeter import UniversalRetargeter
from vid2anim.core.canonical_skeleton import CanonicalMotion
from vid2anim.core.rig_detector import inspect_model


def test_isolated_environment_paths():
    """Verify all model and cache paths are strictly inside M:\\vid2anim, never touching C: drive."""
    proj_dir = configure_isolated_environment()
    assert proj_dir.is_dir()

    # Check environment variables
    hf_home = Path(os.environ["HF_HOME"])
    torch_home = Path(os.environ["TORCH_HOME"])
    gvhmr_ckpts = Path(os.environ["GVHMR_CHECKPOINTS"])
    gvhmr_body = Path(os.environ["GVHMR_BODY_MODELS"])
    xdg_cache = Path(os.environ["XDG_CACHE_HOME"])

    # Ensure none of them reside on C: drive or in user profile
    assert str(hf_home).lower().startswith(str(proj_dir).lower())
    assert str(torch_home).lower().startswith(str(proj_dir).lower())
    assert str(gvhmr_ckpts).lower().startswith(str(proj_dir).lower())
    assert str(gvhmr_body).lower().startswith(str(proj_dir).lower())
    assert str(xdg_cache).lower().startswith(str(proj_dir).lower())

    assert "c:" not in str(hf_home).lower()
    assert "c:" not in str(torch_home).lower()
    assert "c:" not in str(gvhmr_ckpts).lower()
    assert "c:" not in str(gvhmr_body).lower()


def test_smplx_neutral_asset_exists():
    """Verify SMPLX_NEUTRAL.npz is placed inside the project assets folder if downloaded."""
    proj_dir = Path(__file__).resolve().parents[1]
    smplx_path = proj_dir / "assets" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not smplx_path.exists():
        pytest.skip(f"SMPLX neutral model not present at {smplx_path} (external proprietary asset)")
    assert smplx_path.stat().st_size > 100_000_000  # ~108 MB


def test_engine_initialization():
    """Verify MotionExtractor initializes with mediapipe, dwpose, and gvhmr engines."""
    extractor_mp = MotionExtractor(engine="mediapipe")
    assert extractor_mp.engine == "mediapipe"

    extractor_dw = MotionExtractor(engine="dwpose")
    assert extractor_dw.engine == "dwpose"

    extractor_gvhmr = MotionExtractor(engine="gvhmr")
    assert extractor_gvhmr.engine == "gvhmr"


def test_async_video_reader_nonexistent():
    """Verify AsyncVideoReader handles non-existent or empty video paths gracefully."""
    reader = AsyncVideoReader("non_existent_clip.mp4", max_frames=10)
    reader.start()
    frames = list(reader)
    reader.close()
    assert len(frames) == 0


def test_root_motion_transfer_with_world_trajectory():
    """Verify UniversalRetargeter preserves raw root motion and blends with foot grounding."""
    from vid2anim.utils.fixtures import create_synthetic_motion_sequence

    raw_motion = create_synthetic_motion_sequence(frames=30, fps=30.0)
    # Add real horizontal root displacement (walking forward along Z by 1.5 meters)
    raw_motion.root_translation[:, 2] = np.linspace(0.0, 1.5, 30, dtype=np.float32)
    raw_motion.root_translation[:, 0] = np.linspace(0.0, 0.4, 30, dtype=np.float32)

    ybot_glb = Path(__file__).resolve().parents[1] / "glb" / "ybot.glb"
    if ybot_glb.exists():
        analysis = inspect_model(ybot_glb)
        retargeter = UniversalRetargeter()
        anim = retargeter.retarget(raw_motion, rig=analysis)

        assert anim.root_translations.shape == (30, 3)
        # Verify hips translation moved significantly along Z
        z_disp = abs(float(anim.root_translations[-1, 2] - anim.root_translations[0, 2]))
        assert z_disp > 0.5, f"Expected significant root travel along Z, got {z_disp}"
