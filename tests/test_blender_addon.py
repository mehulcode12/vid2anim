"""Tests for the vid2anim Blender Bridge Add-on and download endpoint."""

import json
import urllib.request
import zipfile
from pathlib import Path
import pytest

from plugins.blender.vid2anim_blender import (
    bl_info,
    _encode_multipart_form,
    check_server_status,
)
from vid2anim.server import start_server
import socket
import threading
import time


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def running_server():
    port = find_free_port()
    t = threading.Thread(target=start_server, kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    t.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"


def test_blender_addon_metadata():
    """Verify Blender addon bl_info dictionary complies with Blender 3.0+ requirements."""
    assert bl_info["name"] == "vid2anim Blender Bridge"
    assert bl_info["category"] == "Animation"
    assert len(bl_info["version"]) == 3
    assert bl_info["blender"] >= (3, 0, 0)
    assert "vid2anim" in bl_info["location"]


def test_multipart_form_encoder():
    """Verify standard-library multipart encoder builds valid RFC-compliant form data."""
    fields = {"device": "cpu", "engine": "mediapipe"}
    files = {
        "model": ("test.glb", b"glTF_binary_data", "model/gltf-binary"),
        "media": ("video.mp4", b"dummy_mp4_bytes", "video/mp4"),
    }
    body, ctype = _encode_multipart_form(fields, files)

    assert "multipart/form-data; boundary=" in ctype
    assert b'name="device"\r\n\r\ncpu' in body
    assert b'name="engine"\r\n\r\nmediapipe' in body
    assert b'filename="test.glb"' in body
    assert b"glTF_binary_data" in body
    assert b'filename="video.mp4"' in body
    assert b"dummy_mp4_bytes" in body


def test_blender_addon_download_endpoint(running_server):
    """Verify Web UI endpoint serves the Blender addon script correctly."""
    req = urllib.request.urlopen(f"{running_server}/api/download/blender-addon")
    assert req.status == 200
    assert req.headers.get("Content-Type") == "text/x-python"
    content = req.read().decode("utf-8")
    assert "bl_info" in content
    assert "vid2anim Blender Bridge" in content
    assert "VID2ANIM_OT_AnimateArmature" in content


def test_check_server_status(running_server):
    """Verify addon health checker detects active server."""
    is_online, msg = check_server_status(running_server)
    assert is_online is True
    assert "Server Online" in msg

    # Bad port
    is_bad, msg_bad = check_server_status("http://127.0.0.1:59999")
    assert is_bad is False
    assert "Offline" in msg_bad


def test_exported_addon_zip_bl_info(tmp_path):
    """Verify exported addon zip contains valid bl_info in __init__.py for Blender AST parser."""
    from vid2anim.utils.cli_features import handle_addon_command
    zip_path = tmp_path / "test_addon.zip"
    handle_addon_command(export_path=str(zip_path), auto_install=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        init_code = zf.read("vid2anim_blender/__init__.py").decode("utf-8")
        assert "bl_info = {" in init_code
        assert '"name": "vid2anim Blender Bridge"' in init_code

