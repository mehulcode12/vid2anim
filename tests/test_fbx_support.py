from pathlib import Path
import pytest
from vid2anim.utils.fbx_support import convert_fbx_to_glb, ensure_glb_model
from vid2anim.core.rig_detector import inspect_model, RigType


def test_fbx_support_with_remy_model():
    fbx_file = Path("glb/Remy.fbx")
    if not fbx_file.exists():
        pytest.skip("glb/Remy.fbx not found")

    out_glb = Path("scratch/test_unit_remy.glb")
    if out_glb.exists():
        out_glb.unlink()

    converted = convert_fbx_to_glb(fbx_file, output_glb_path=out_glb)
    assert converted.exists()
    assert converted.stat().st_size > 0

    analysis = inspect_model(converted)
    assert analysis.is_rigged
    assert analysis.rig_type == RigType.MIXAMO
    assert len(analysis.bone_mapping) >= 15

    # Test ensure_glb_model on existing GLB (should return unchanged)
    ensured = ensure_glb_model(converted)
    assert ensured == converted

    # Clean up
    if out_glb.exists():
        out_glb.unlink()
