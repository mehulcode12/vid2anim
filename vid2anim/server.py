"""Zero-Dependency Web UI & API Server for vid2anim.

Provides a modern browser interface with drag-and-drop file upload, real-time
progress updates, and an interactive Three.js 3D model viewport.
Runs using Python's standard library http.server without requiring external web frameworks.
"""

from __future__ import annotations

import email
import email.policy
import http.server
import json
import os
import socketserver
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict

from .core.rig_detector import inspect_model
from .pipeline import PipelineConfig, Vid2AnimPipeline
from .utils.diagnostics import get_system_diagnostics


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vid2anim - Animate Any 3D Character</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --accent: #58a6ff;
      --accent-hover: #1f6feb;
      --text: #c9d1d9;
      --text-bright: #ffffff;
      --success: #238636;
      --warning: #d29922;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card);
    }
    header h1 {
      font-size: 1.25rem;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 12px;
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
    }
    .main-container {
      display: grid;
      grid-template-columns: 380px 1fr;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      padding: 24px;
      border-right: 1px solid var(--border);
      background: var(--card);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(255, 255, 255, 0.02);
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(88, 166, 255, 0.05);
    }
    .upload-zone p { font-size: 0.85rem; margin-top: 6px; color: #8b949e; }
    .file-name { font-size: 0.8rem; color: var(--accent); margin-top: 6px; word-break: break-all; }
    .input-group { display: flex; flex-direction: column; gap: 8px; }
    .input-group label { font-size: 0.85rem; font-weight: 600; color: var(--text-bright); }
    select, button {
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #21262d;
      color: var(--text-bright);
      font-size: 0.9rem;
      cursor: pointer;
    }
    button.primary {
      background: var(--success);
      border-color: rgba(255, 255, 255, 0.1);
      font-weight: 600;
      transition: background 0.2s;
    }
    button.primary:hover:not(:disabled) { background: #2ea043; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-card {
      padding: 12px;
      border-radius: 6px;
      background: #21262d;
      border: 1px solid var(--border);
      font-size: 0.85rem;
      line-height: 1.4;
    }
    .progress-bar-bg {
      height: 8px;
      background: #30363d;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.3s;
    }
    .viewport-container {
      position: relative;
      background: #05070a;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #viewport3d { width: 100%; height: 100%; }
    .overlay-controls {
      position: absolute;
      bottom: 20px;
      right: 20px;
      display: flex;
      gap: 10px;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .empty-state {
      position: absolute;
      text-align: center;
      color: #8b949e;
      pointer-events: none;
    }
    .empty-state h3 { color: var(--text-bright); margin-bottom: 6px; }
  </style>
</head>
<body>
  <header>
    <h1>vid2anim <span>AI Motion Retargeter</span></h1>
    <div id="deviceBadge" class="badge">Loading device...</div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <!-- Media Upload -->
      <div class="input-group">
        <label>1. Input Video or Photo (Max 30s)</label>
        <div class="upload-zone" onclick="document.getElementById('mediaInput').click()">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          <p>Drop MP4 video or JPG/PNG photo here</p>
          <div id="mediaName" class="file-name"></div>
        </div>
        <input type="file" id="mediaInput" accept="video/mp4,video/quicktime,image/jpeg,image/png" style="display:none">
      </div>

      <!-- 3D Model Upload -->
      <div class="input-group">
        <label>2. Target 3D Model</label>
        <div class="upload-zone" onclick="document.getElementById('modelInput').click()">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
          <p>Drop .GLB, .FBX, or .OBJ character</p>
          <div id="modelName" class="file-name"></div>
        </div>
        <input type="file" id="modelInput" accept=".glb,.gltf,.fbx,.obj" style="display:none">
      </div>

      <!-- Device Setting -->
      <div class="input-group">
        <label>3. Compute Engine</label>
        <select id="deviceSelect">
          <option value="cpu" selected>CPU (Recommended / Lightweight)</option>
          <option value="cuda">GPU (NVIDIA CUDA)</option>
        </select>
      </div>

      <!-- Action Button -->
      <button id="runBtn" class="primary" onclick="startAnimationPipeline()" disabled>Animate 3D Model</button>

      <!-- Status -->
      <div id="statusCard" class="status-card" style="display:none">
        <div id="statusText">Ready</div>
        <div class="progress-bar-bg">
          <div id="progressBar" class="progress-bar-fill"></div>
        </div>
      </div>

      <button id="downloadBtn" onclick="downloadResult()" style="display:none">Download Animated .GLB</button>
    </div>

    <!-- 3D Viewport -->
    <div class="viewport-container">
      <div id="emptyState" class="empty-state">
        <h3>3D Character Viewport</h3>
        <p>Upload a model and video to preview your animated character here.</p>
      </div>
      <div id="viewport3d"></div>
      <div class="overlay-controls" id="playbackControls" style="display:none">
        <select id="animSelect" onchange="switchAnimation(this.value)" style="display:none; background:#161b22; color:#c9d1d9; border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:12px;"></select>
        <button onclick="togglePlayback()">Play / Pause</button>
        <button onclick="resetCamera()">Reset View</button>
      </div>
    </div>
  </div>

  <script>
    let scene, camera, renderer, controls, mixer, clock, currentAction;
    let selectedMedia = null, selectedModel = null, lastResultBlobUrl = null;
    let currentGltfClips = [];

    // Init Three.js Scene
    function init3D() {
      const container = document.getElementById('viewport3d');
      scene = new THREE.Scene();
      clock = new THREE.Clock();

      camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
      camera.position.set(0, 1.4, 3.5);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.outputEncoding = THREE.sRGBEncoding;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0, 1.0, 0);
      controls.update();

      const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
      scene.add(hemiLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
      dirLight.position.set(2, 4, 3);
      scene.add(dirLight);

      const grid = new THREE.GridHelper(10, 20, 0x30363d, 0x161b22);
      scene.add(grid);

      window.addEventListener('resize', onWindowResize);
      animate();
    }

    function onWindowResize() {
      const container = document.getElementById('viewport3d');
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }

    function animate() {
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      if (mixer) mixer.update(delta);
      controls.update();
      renderer.render(scene, camera);
    }

    // Load GLB into viewport
    function loadGLBModel(url) {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('playbackControls').style.display = 'flex';

      const loader = new THREE.GLTFLoader();
      loader.load(url, (gltf) => {
        // Clear previous mixer and scene objects
        if (mixer) {
          mixer.stopAllAction();
        }
        scene.children.filter(c => c.isGroup || c.isMesh).forEach(c => scene.remove(c));

        const model = gltf.scene;
        scene.add(model);

        currentGltfClips = gltf.animations || [];
        if (currentGltfClips.length > 0) {
          mixer = new THREE.AnimationMixer(model);

          // Populate clip switcher
          const animSelect = document.getElementById('animSelect');
          if (animSelect) {
            animSelect.innerHTML = '';
            currentGltfClips.forEach((clip, idx) => {
              const opt = document.createElement('option');
              opt.value = idx;
              opt.textContent = (clip.name || `Track ${idx + 1}`) + ` (${clip.duration.toFixed(1)}s)`;
              animSelect.appendChild(opt);
            });
            animSelect.style.display = currentGltfClips.length > 1 ? 'inline-block' : 'none';
          }

          // Prioritize newly retargeted Vid2Anim_Motion clip
          const targetClip = currentGltfClips.find(a => a.name === 'Vid2Anim_Motion') || currentGltfClips[0];
          const targetIdx = currentGltfClips.indexOf(targetClip);
          if (animSelect && targetIdx >= 0) {
            animSelect.value = targetIdx;
          }

          currentAction = mixer.clipAction(targetClip);
          currentAction.play();
        }
      });
    }

    function switchAnimation(idx) {
      if (!mixer || !currentGltfClips[idx]) return;
      mixer.stopAllAction();
      currentAction = mixer.clipAction(currentGltfClips[idx]);
      currentAction.play();
    }

    function togglePlayback() {
      if (!currentAction) return;
      currentAction.paused = !currentAction.paused;
    }

    function resetCamera() {
      camera.position.set(0, 1.4, 3.5);
      controls.target.set(0, 1.0, 0);
      controls.update();
    }

    // Diagnostics & File Selection
    fetch('/api/diagnostics')
      .then(r => r.json())
      .then(d => {
        const badge = document.getElementById('deviceBadge');
        badge.innerText = d.cuda_available ? 'GPU Mode (CUDA Available)' : 'CPU Mode (Optimized)';
      });

    document.getElementById('mediaInput').addEventListener('change', (e) => {
      if (e.target.files[0]) {
        selectedMedia = e.target.files[0];
        document.getElementById('mediaName').innerText = selectedMedia.name;
        checkReady();
      }
    });

    document.getElementById('modelInput').addEventListener('change', (e) => {
      if (e.target.files[0]) {
        selectedModel = e.target.files[0];
        document.getElementById('modelName').innerText = selectedModel.name;
        // Preview local model in 3D right away
        const url = URL.createObjectURL(selectedModel);
        loadGLBModel(url);
        checkReady();
      }
    });

    function checkReady() {
      document.getElementById('runBtn').disabled = !(selectedMedia && selectedModel);
    }

    async function startAnimationPipeline() {
      const btn = document.getElementById('runBtn');
      const statusCard = document.getElementById('statusCard');
      const statusText = document.getElementById('statusText');
      const progressBar = document.getElementById('progressBar');

      btn.disabled = true;
      statusCard.style.display = 'block';
      statusText.innerText = 'Uploading assets and extracting 3D motion...';
      progressBar.style.width = '20%';

      const formData = new FormData();
      formData.append('media', selectedMedia);
      formData.append('model', selectedModel);
      formData.append('device', document.getElementById('deviceSelect').value);

      try {
        progressBar.style.width = '45%';
        statusText.innerText = 'Retargeting kinematic motion onto character...';

        const response = await fetch('/api/animate', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.error || 'Animation pipeline failed');
        }

        progressBar.style.width = '90%';
        statusText.innerText = 'Packing and loading animated character...';

        const blob = await response.blob();
        lastResultBlobUrl = URL.createObjectURL(blob);
        loadGLBModel(lastResultBlobUrl);

        progressBar.style.width = '100%';
        statusText.innerText = 'Animation complete! Playing in 3D viewport.';
        document.getElementById('downloadBtn').style.display = 'block';
      } catch (err) {
        statusText.innerText = 'Error: ' + err.message;
        progressBar.style.width = '0%';
      } finally {
        btn.disabled = false;
      }
    }

    function downloadResult() {
      if (!lastResultBlobUrl) return;
      const a = document.createElement('a');
      a.href = lastResultBlobUrl;
      a.download = (selectedModel ? selectedModel.name.replace(/\.[^/.]+$/, '') : 'character') + '_animated.glb';
      a.click();
    }

    window.onload = init3D;
  </script>
</body>
</html>
"""


class Vid2AnimHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Handles API requests and serves the interactive 3D Web UI."""

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif parsed_path.path == "/api/diagnostics":
            diag = get_system_diagnostics()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(diag).encode("utf-8"))

        elif parsed_path.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "vid2anim"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/api/animate":
            self._handle_animate_post()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_animate_post(self) -> None:
        """Parse uploaded media and 3D model, run pipeline, and stream back result."""
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Expected multipart/form-data"}).encode("utf-8"))
                return

            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)

            # Parse multipart using standard email policy
            raw_msg = (
                b"Content-Type: "
                + content_type.encode("latin-1")
                + b"\r\n\r\n"
                + body_bytes
            )
            msg = email.message_from_bytes(raw_msg, policy=email.policy.default)

            fields: Dict[str, Any] = {}
            for part in msg.iter_parts():
                cd = part.get("Content-Disposition", "")
                name = None
                filename = part.get_filename()
                for param in cd.split(";"):
                    param = param.strip()
                    if param.startswith("name="):
                        name = param[5:].strip('"')
                        break
                if name:
                    payload = part.get_payload(decode=True)
                    if filename:
                        fields[name] = {"filename": filename, "data": payload}
                    else:
                        fields[name] = payload.decode("utf-8", errors="ignore")

            if "media" not in fields or "model" not in fields:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'media' or 'model' field in upload"}).encode("utf-8"))
                return

            media_info = fields["media"]
            model_info = fields["model"]
            device_str = fields.get("device", "auto")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir = Path(tmpdir)
                media_path = tmp_dir / (media_info.get("filename") or "input_media.mp4")
                model_path = tmp_dir / (model_info.get("filename") or "character.glb")
                output_glb = tmp_dir / "animated_character.glb"

                # Write uploaded bytes to temporary files
                media_path.write_bytes(media_info["data"])
                model_path.write_bytes(model_info["data"])

                # Execute pipeline
                config = PipelineConfig(
                    input_media=media_path,
                    target_model=model_path,
                    output_glb=output_glb,
                    device=device_str,
                    max_duration_sec=30.0,
                    smooth=True,
                )
                pipeline = Vid2AnimPipeline(config)
                result = pipeline.execute()

                if not result.success:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": result.error_message or "Execution failed"}).encode("utf-8"))
                    return

                # Send back the generated GLB binary
                glb_data = output_glb.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(glb_data)))
                self.end_headers()
                self.wfile.write(glb_data)

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the built-in local web server."""
    with socketserver.TCPServer((host, port), Vid2AnimHTTPHandler) as httpd:
        print("=" * 60)
        print(f" vid2anim Web UI & API Server running at:")
        print(f"   http://{host}:{port}")
        print(" Press Ctrl+C to stop.")
        print("=" * 60)
        httpd.serve_forever()
