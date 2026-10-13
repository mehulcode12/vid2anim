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
