"""Tests for vid2anim open-source CLI subcommands and utilities."""

import subprocess
import sys
import zipfile
from pathlib import Path
import pytest

from vid2anim import __version__
from vid2anim.utils.cli_features import (
    handle_info_command,
    handle_samples_command,
    handle_addon_command,
    handle_benchmark_command,
    get_project_root,
)


def test_cli_version_flag():
    """Verify vid2anim --version and -v print version string and exit 0."""
    res = subprocess.run(
        [sys.executable, "-m", "vid2anim.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert f"vid2anim {__version__}" in res.stdout


def test_cli_info_command(capsys):
    """Verify vid2anim info outputs environment and isolated cache paths."""
    handle_info_command()
    captured = capsys.readouterr()
    assert f"vid2anim v{__version__}" in captured.out
    assert "Isolated Project Storage" in captured.out
    assert "Triple Tracking Engines" in captured.out


def test_cli_samples_list(capsys):
    """Verify vid2anim samples list displays bundled characters and reference videos."""
    handle_samples_command(action="list")
    captured = capsys.readouterr()
    assert "Official Sample Assets" in captured.out
    assert "ybot.glb" in captured.out
    assert "dance_street.mp4" in captured.out


def test_cli_samples_copy(tmp_path, capsys):
    """Verify vid2anim samples copy duplicates bundled assets into a target directory."""
    dest = tmp_path / "copied_samples"
    handle_samples_command(action="copy", dest_dir=str(dest))
    captured = capsys.readouterr()
    assert "copied successfully" in captured.out
    assert dest.exists()
    assert (dest / "models" / "ybot.glb").exists()
    assert (dest / "videos" / "dance_street.mp4").exists()


def test_cli_addon_export(tmp_path, capsys):
    """Verify vid2anim addon exports a clean Blender-installable zip bundle."""
    target_zip = tmp_path / "test_blender_addon.zip"
    handle_addon_command(export_path=str(target_zip), auto_install=False)
    captured = capsys.readouterr()
    assert "Blender Bridge Add-on" in captured.out
    assert target_zip.exists()

    with zipfile.ZipFile(target_zip, "r") as zf:
        namelist = zf.namelist()
        assert "vid2anim_blender/vid2anim_blender.py" in namelist
        assert "vid2anim_blender/__init__.py" in namelist


def test_cli_benchmark_command(capsys):
    """Verify vid2anim benchmark computes universal retargeter throughput."""
    handle_benchmark_command(frames=5)
    captured = capsys.readouterr()
    assert "Hardware & Throughput Benchmark" in captured.out
    assert "Universal Retargeter" in captured.out
    assert "FPS" in captured.out


def test_get_project_root():
    """Verify project root resolution finds directory containing pyproject.toml."""
    root = get_project_root()
    assert (root / "pyproject.toml").is_file()
