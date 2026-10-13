"""Test suite for vid2anim Web UI and REST API server."""

import io
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path
import pytest

from vid2anim.server import start_server


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def running_server():
    port = find_free_port()
    t = threading.Thread(target=start_server, kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    t.start()
    time.sleep(0.3)  # Wait for socket to bind
    yield f"http://127.0.0.1:{port}"


def test_server_homepage_html(running_server):
    req = urllib.request.urlopen(f"{running_server}/")
    assert req.status == 200
    content = req.read().decode("utf-8")
    assert "vid2anim" in content
    assert "three.min.js" in content
    assert "viewport3d" in content
    # Dual-Engine Model Selection UI
    assert "engineCardMediapipe" in content
    assert "engineCardDwpose" in content
    assert "DWPose WholeBody" in content
    assert "selectedEngine" in content
    # Rig Specifications Guide Modal
    assert "rigGuideModal" in content
    assert "openRigGuideModal" in content
    assert "Supported Formats & Rigs" in content
    assert "Blender Add-on" in content


def test_server_diagnostics_endpoint(running_server):
    req = urllib.request.urlopen(f"{running_server}/api/diagnostics")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "cpu_count" in data
    assert "cuda_available" in data
    assert "recommended_device" in data


def test_server_health_check(running_server):
    req = urllib.request.urlopen(f"{running_server}/api/health")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["status"] == "ok"


def test_server_progress_endpoint(running_server):
    req = urllib.request.urlopen(f"{running_server}/api/progress")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "active" in data
    assert "percent" in data
    assert "step" in data
    assert "message" in data


def test_server_empty_upload_rejected(running_server):
    """Test that POST /api/animate rejects invalid non-multipart or empty uploads with 400."""
    try:
        req = urllib.request.Request(
            f"{running_server}/api/animate",
            data=b"invalid_raw_body",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req)
        assert False, "Should have raised HTTPError 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400


