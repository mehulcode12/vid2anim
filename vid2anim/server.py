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
from .utils.progress import ProgressReporter


PIPELINE_PROGRESS: Dict[str, Any] = {
    "active": False,
    "percent": 0,
    "step": "Idle",
    "message": "Ready",
}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vid2anim - 3D Character Motion Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
  <style>
    :root {
      --bg-dark: #07090e;
      --bg-panel: #0d121f;
      --bg-card: #141b2d;
      --bg-card-hover: #1a233a;
      --border: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(99, 102, 241, 0.35);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.3);
      --accent-grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
      --cyan: #06b6d4;
      --emerald: #10b981;
      --amber: #f59e0b;
      --text: #cbd5e1;
      --text-bright: #ffffff;
      --text-dim: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-dark);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      user-select: none;
    }
    header {
      padding: 12px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-panel);
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: var(--accent-grad);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-pill {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .badge-device {
      background: rgba(16, 185, 129, 0.12);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-sync {
      background: rgba(6, 182, 212, 0.12);
      color: #38bdf8;
      border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .badge-sync.active {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
      border-color: rgba(16, 185, 129, 0.4);
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }
    .main-container {
      display: grid;
      grid-template-columns: 350px 1fr;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      padding: 20px;
      border-right: 1px solid var(--border);
      background: var(--bg-panel);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .input-group label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-bright);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
    }
    .upload-zone {
      border: 1.5px dashed var(--border);
      border-radius: 10px;
      padding: 16px 12px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      background: rgba(255, 255, 255, 0.015);
      position: relative;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.06);
      transform: translateY(-1px);
    }
    .upload-zone svg {
      stroke: #818cf8;
      margin-bottom: 4px;
    }
    .upload-zone p { font-size: 0.8rem; color: var(--text-dim); }
    .file-name {
      font-size: 0.78rem;
      color: #a5b4fc;
      margin-top: 6px;
      word-break: break-all;
      font-weight: 500;
    }
    select, button {
      font-family: inherit;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      color: var(--text-bright);
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    select:focus, button:focus {
      outline: none;
      border-color: var(--accent);
    }
    button.primary {
      background: var(--accent-grad);
      border: none;
      font-weight: 600;
      font-size: 0.92rem;
      padding: 12px;
      box-shadow: 0 4px 16px var(--accent-glow);
    }
    button.primary:hover:not(:disabled) {
      filter: brightness(1.1);
      transform: translateY(-1px);
    }
    .engine-cards {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .engine-card {
      padding: 10px 12px;
      border-radius: 8px;
      background: var(--bg-card);
      border: 1.5px solid var(--border);
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .engine-card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      background: var(--bg-card-hover);
    }
    .engine-card.active {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.12);
      box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
    }
    .engine-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }
    .engine-title {
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .engine-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-dim);
    }
    .engine-card.active .engine-dot {
      background: #6366f1;
      box-shadow: 0 0 6px #6366f1;
    }
    .engine-tag {
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .tag-fast {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-precision {
      background: rgba(139, 92, 246, 0.18);
      color: #c084fc;
      border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .engine-desc {
      font-size: 0.73rem;
      color: var(--text-dim);
      line-height: 1.35;
    }
    .status-card {
      padding: 14px;
      border-radius: 8px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      font-size: 0.82rem;
      line-height: 1.4;
    }
    .status-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 6px;
      font-weight: 600;
      color: var(--text-bright);
    }
    .progress-bar-bg {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--accent-grad);
      width: 0%;
      transition: width 0.3s ease;
    }
    .download-btn {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .download-btn:hover {
      background: rgba(16, 185, 129, 0.25);
    }

    /* Stage Area */
    .stage-area {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      background: #04060a;
    }
    .stage-header {
      padding: 8px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 5;
    }
    .view-mode-tabs {
      display: flex;
      background: var(--bg-card);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--border);
      gap: 2px;
    }
    .view-tab {
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 500;
      border: none;
      background: transparent;
      color: var(--text-dim);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .view-tab.active {
      background: var(--accent);
      color: #ffffff;
      box-shadow: 0 2px 8px var(--accent-glow);
    }
    .stage-tools {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Viewports Layout Container */
    .viewports-container {
      flex: 1;
      display: grid;
      overflow: hidden;
      position: relative;
    }
    .viewports-container.mode-split {
      grid-template-columns: 1fr 1fr;
    }
    .viewports-container.mode-3d {
      grid-template-columns: 1fr;
    }
    .viewports-container.mode-3d #videoViewport {
      display: none;
    }
    .viewports-container.mode-video {
      grid-template-columns: 1fr;
    }
    .viewports-container.mode-video #modelViewport {
      display: none;
    }

    .viewport-card {
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--border);
      background: #050811;
    }
    .viewport-card:last-child {
      border-right: none;
    }
    .viewport-tag {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 4;
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(13, 18, 31, 0.85);
      backdrop-filter: blur(8px);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: #cbd5e1;
    }
    .viewport-meta {
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.08);
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Video player container */
    #videoViewport {
      align-items: center;
      justify-content: center;
      background: #020307;
    }
    #sourceVideo {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: transparent;
      outline: none;
    }
    #sourceImage {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
    }

    /* 3D Viewport container */
    #modelViewport {
      background: radial-gradient(circle at center, #0e1526 0%, #050811 100%);
    }
    #viewport3d {
      width: 100%;
      height: 100%;
      display: block;
    }
    .viewport-hints {
      position: absolute;
      bottom: 12px;
      right: 12px;
      z-index: 4;
      font-size: 0.68rem;
      color: rgba(255, 255, 255, 0.35);
      background: rgba(7, 9, 14, 0.7);
      backdrop-filter: blur(4px);
      padding: 4px 8px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      pointer-events: none;
    }

    /* Empty states */
    .viewport-empty {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: var(--text-dim);
      padding: 24px;
      pointer-events: none;
      z-index: 2;
    }
    .viewport-empty svg {
      stroke: rgba(255, 255, 255, 0.15);
      margin-bottom: 12px;
    }
    .viewport-empty h4 {
      font-size: 0.95rem;
      color: var(--text);
      margin-bottom: 4px;
    }
    .viewport-empty p {
      font-size: 0.78rem;
      max-width: 220px;
    }

    /* Bottom Docked Playback Bar */
    .sync-controller {
      background: var(--bg-panel);
      border-top: 1px solid var(--border);
      padding: 10px 20px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 6;
    }
    .timeline-container {
      display: flex;
      align-items: center;
      gap: 12px;
      position: relative;
    }
    .time-slider {
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 6px;
      border-radius: 999px;
      background: #1e293b;
      outline: none;
      cursor: pointer;
      position: relative;
    }
    .time-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #818cf8;
      box-shadow: 0 0 10px #6366f1;
      cursor: pointer;
      border: 2px solid #ffffff;
      transition: transform 0.1s;
    }
    .time-slider::-webkit-slider-thumb:hover {
      transform: scale(1.25);
    }
    .controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ctrl-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .ctrl-btn {
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-bright);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .ctrl-btn:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.15);
    }
    .ctrl-btn.play-btn {
      background: var(--accent);
      border-color: var(--accent);
      padding: 6px 16px;
      font-weight: 600;
    }
    .ctrl-btn.play-btn:hover {
      background: var(--accent-hover);
    }
    .ctrl-btn.active {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent);
      color: #a5b4fc;
    }
    .time-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 6px;
    }
    .frame-pill {
      font-size: 0.72rem;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.05);
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        vid2anim
      </div>
      <div class="brand-pill">Universal Retargeting Studio</div>
    </div>
    <div class="header-actions">
      <div id="syncBadge" class="badge badge-sync active">
        <span class="badge-dot"></span>
        <span id="syncText">Dual-Viewport Synced</span>
      </div>
      <div id="deviceBadge" class="badge badge-device">Loading device...</div>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <!-- Media Upload -->
      <div class="input-group">
        <label>1. Source Footage <span>Max 30s</span></label>
        <div class="upload-zone" onclick="document.getElementById('mediaInput').click()">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          <p>Drop MP4 / MOV video or photo</p>
          <div id="mediaName" class="file-name"></div>
        </div>
        <input type="file" id="mediaInput" accept="video/mp4,video/quicktime,video/webm,image/jpeg,image/png" style="display:none">
      </div>

      <!-- 3D Model Upload -->
      <div class="input-group">
        <label>2. Target 3D Model <span>Rigged Mesh</span></label>
        <div class="upload-zone" onclick="document.getElementById('modelInput').click()">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
          <p>Drop .GLB, .FBX, or .OBJ character</p>
          <div id="modelName" class="file-name"></div>
        </div>
        <input type="file" id="modelInput" accept=".glb,.gltf,.fbx,.obj" style="display:none">
      </div>

      <!-- Tracking Engine Setting -->
      <div class="input-group">
        <label>3. Tracking AI Model <span>Dual-Engine</span></label>
        <div class="engine-cards">
          <div class="engine-card active" id="engineCardMediapipe" onclick="selectEngine('mediapipe')">
            <div class="engine-card-header">
              <div class="engine-title">
                <span class="engine-dot"></span>
                MediaPipe Pose
              </div>
              <span class="engine-tag tag-fast">Fast CPU</span>
            </div>
            <p class="engine-desc">30+ FPS real-time CPU tracking. Instant startup, lightweight memory footprint.</p>
          </div>
          <div class="engine-card" id="engineCardDwpose" onclick="selectEngine('dwpose')">
            <div class="engine-card-header">
              <div class="engine-title">
                <span class="engine-dot"></span>
                DWPose WholeBody
              </div>
              <span class="engine-tag tag-precision">133 Pts Deep AI</span>
            </div>
            <p class="engine-desc">133 whole-body keypoints + full 21-joint per-hand finger tracking via ONNX.</p>
          </div>
        </div>
        <input type="hidden" id="selectedEngine" value="mediapipe">
      </div>

      <!-- Device Setting -->
      <div class="input-group">
        <label>4. Compute Hardware</label>
        <select id="deviceSelect">
          <option value="cpu" selected>CPU (Lightweight / Stable)</option>
          <option value="cuda">GPU (NVIDIA CUDA Acceleration)</option>
        </select>
      </div>

      <!-- Action Button -->
      <button id="runBtn" class="primary" onclick="startAnimationPipeline()" disabled>Animate 3D Model</button>

      <!-- Status Card -->
      <div id="statusCard" class="status-card" style="display:none">
        <div class="status-header">
          <span id="statusTitle">Processing</span>
          <span id="statusPercent" style="color:var(--cyan)">0%</span>
        </div>
        <div id="statusText">Ready</div>
        <div class="progress-bar-bg">
          <div id="progressBar" class="progress-bar-fill"></div>
        </div>
      </div>

      <button id="downloadBtn" class="download-btn" onclick="downloadResult()" style="display:none">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download Animated .GLB
      </button>
    </div>

    <!-- Main Stage Viewport Area -->
    <div class="stage-area">
      <div class="stage-header">
        <div class="view-mode-tabs">
          <button class="view-tab active" id="tabSplit" onclick="setViewMode('split')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"/><rect x="13" y="3" width="8" height="18" rx="2"/></svg>
            Side-by-Side
          </button>
          <button class="view-tab" id="tab3d" onclick="setViewMode('3d')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
            3D Focus
          </button>
          <button class="view-tab" id="tabVideo" onclick="setViewMode('video')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Video Only
          </button>
        </div>

        <div class="stage-tools">
          <select id="animSelect" onchange="switchAnimation(this.value)" style="display:none; font-size:0.75rem; padding:4px 8px;"></select>
          <button class="ctrl-btn" onclick="resetCamera()" title="Reset 3D Camera">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
            Reset Cam
          </button>
        </div>
      </div>

      <!-- Viewports Grid -->
      <div class="viewports-container mode-split" id="viewportsContainer">
        <!-- Video Viewport -->
        <div class="viewport-card" id="videoViewport">
          <div class="viewport-tag">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Original Video
            <span id="videoMetaBadge" class="viewport-meta">No media</span>
          </div>
          <div id="videoEmptyState" class="viewport-empty">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
            <h4>Reference Footage</h4>
            <p>Upload a video on the left to preview original motion here</p>
          </div>
          <video id="sourceVideo" playsinline muted style="display:none"></video>
          <img id="sourceImage" alt="Reference Image">
        </div>

        <!-- 3D Viewport -->
        <div class="viewport-card" id="modelViewport">
          <div class="viewport-tag">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            Retargeted 3D Mesh
            <span id="modelMetaBadge" class="viewport-meta">Waiting for model</span>
          </div>
          <div id="modelEmptyState" class="viewport-empty">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            <h4>3D Character Viewport</h4>
            <p>Upload a .GLB/.FBX character to preview retargeted animations</p>
          </div>
          <div id="viewport3d"></div>
          <div class="viewport-hints">Left Click: Rotate | Right Click: Pan | Scroll: Zoom</div>
        </div>
      </div>

      <!-- Bottom Docked Synchronization Controller -->
      <div class="sync-controller" id="syncController">
        <div class="timeline-container">
          <input type="range" id="timeSlider" class="time-slider" min="0" max="1" step="0.01" value="0">
        </div>

        <div class="controls-row">
          <!-- Playback Controls -->
          <div class="ctrl-group">
            <button id="playBtn" class="ctrl-btn play-btn" onclick="togglePlayback()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              <span>Play</span>
            </button>
            <button class="ctrl-btn" onclick="stepFrame(-1)" title="Previous Frame (Left Arrow)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
              -1 Frame
            </button>
            <button class="ctrl-btn" onclick="stepFrame(1)" title="Next Frame (Right Arrow)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
              +1 Frame
            </button>
            <div class="time-display">
              <span id="timecodeText">00:00.00 / 00:00.00</span>
              <span id="framePill" class="frame-pill">Frame 0 / 0</span>
            </div>
          </div>

          <!-- Speed & Loop -->
          <div class="ctrl-group">
            <div style="display:flex; gap:2px; background:var(--bg-card); padding:2px; border-radius:6px; border:1px solid var(--border)">
              <button class="ctrl-btn" style="padding:3px 8px; font-size:0.75rem;" onclick="setSpeed(0.25)">0.25x</button>
              <button class="ctrl-btn" style="padding:3px 8px; font-size:0.75rem;" onclick="setSpeed(0.5)">0.5x</button>
              <button class="ctrl-btn active" id="spd1x" style="padding:3px 8px; font-size:0.75rem;" onclick="setSpeed(1.0)">1.0x</button>
              <button class="ctrl-btn" style="padding:3px 8px; font-size:0.75rem;" onclick="setSpeed(1.5)">1.5x</button>
              <button class="ctrl-btn" style="padding:3px 8px; font-size:0.75rem;" onclick="setSpeed(2.0)">2.0x</button>
            </div>
            <button id="loopBtn" class="ctrl-btn active" onclick="toggleLoop()" title="Toggle Looping">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
              Loop
            </button>
            <button id="muteBtn" class="ctrl-btn active" onclick="toggleMute()" title="Toggle Audio">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
              Muted
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let scene, camera, renderer, controls, mixer, clock, currentAction;
    let selectedMedia = null, selectedModel = null, lastResultBlobUrl = null;
    let currentGltfClips = [];

    // Master Synchronization State
    const sourceVideo = document.getElementById('sourceVideo');
    const sourceImage = document.getElementById('sourceImage');
    const timeSlider = document.getElementById('timeSlider');
    const playBtn = document.getElementById('playBtn');
    const timecodeText = document.getElementById('timecodeText');
    const framePill = document.getElementById('framePill');
    const videoMetaBadge = document.getElementById('videoMetaBadge');
    const modelMetaBadge = document.getElementById('modelMetaBadge');
    const syncBadge = document.getElementById('syncBadge');
    const syncText = document.getElementById('syncText');

    let isPlaying = false;
    let isSeeking = false;
    let isLooping = true;
    let playbackSpeed = 1.0;
    let effectiveFps = 25.0;
    let totalDuration = 0.0;
    let lastAnimatedBoneCount = 0;

    // Initialize Three.js Viewport
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

      const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
      dirLight.position.set(2, 4, 3);
      scene.add(dirLight);

      const backLight = new THREE.DirectionalLight(0x6366f1, 0.8);
      backLight.position.set(-2, 2, -2);
      scene.add(backLight);

      const grid = new THREE.GridHelper(10, 20, 0x30363d, 0x161b22);
      scene.add(grid);

      window.addEventListener('resize', onWindowResize);
      if (window.ResizeObserver) {
        new ResizeObserver(onWindowResize).observe(container);
      }

      setupKeyboardShortcuts();
      setupVideoEvents();
      animate();
    }

    function onWindowResize() {
      const container = document.getElementById('viewport3d');
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }

    // Main Render & Synchronizer Loop
    function animate() {
      requestAnimationFrame(animate);

      if (isPlaying && !isSeeking) {
        let curTime = 0;

        // If source video exists, use it as Master Hardware Clock
        if (sourceVideo && sourceVideo.src && !sourceVideo.paused && sourceVideo.readyState >= 2) {
          curTime = sourceVideo.currentTime;
          if (mixer && currentAction) {
            const clipDur = currentAction.getClip().duration;
            const animTime = clipDur > 0 ? (curTime % clipDur) : 0;
            mixer.setTime(animTime);
          }
        } else if (mixer && currentAction && !currentAction.paused) {
          // Fallback to clock delta if 3D only
          const delta = clock.getDelta();
          mixer.update(delta * playbackSpeed);
          const clipDur = currentAction.getClip().duration || 1;
          curTime = mixer.time % clipDur;
        }

        timeSlider.value = curTime;
        updateTimeDisplays(curTime);
      }

      controls.update();
      renderer.render(scene, camera);
    }

    // Video Events & Sync
    function setupVideoEvents() {
      sourceVideo.addEventListener('loadedmetadata', () => {
        totalDuration = Math.max(totalDuration, sourceVideo.duration);
        timeSlider.max = totalDuration;
        videoMetaBadge.innerText = `${sourceVideo.videoWidth}×${sourceVideo.videoHeight} • ${sourceVideo.duration.toFixed(1)}s`;
        updateTimeDisplays(0);
        syncBadge.className = 'badge badge-sync active';
        syncText.innerText = 'Lockstep Sync Active';
      });

      sourceVideo.addEventListener('ended', () => {
        if (isLooping) {
          seekTo(0);
          play();
        } else {
          pause();
        }
      });
    }

    // View Modes
    function setViewMode(mode) {
      document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
      const container = document.getElementById('viewportsContainer');
      container.className = 'viewports-container mode-' + mode;

      if (mode === 'split') document.getElementById('tabSplit').classList.add('active');
      else if (mode === '3d') document.getElementById('tab3d').classList.add('active');
      else if (mode === 'video') document.getElementById('tabVideo').classList.add('active');

      setTimeout(onWindowResize, 30);
    }

    // GLB Loading
    function loadGLBModel(url) {
      document.getElementById('modelEmptyState').style.display = 'none';

      const loader = new THREE.GLTFLoader();
      loader.load(url, (gltf) => {
        if (mixer) mixer.stopAllAction();
        scene.children.filter(c => c.isGroup || c.isMesh).forEach(c => scene.remove(c));

        const model = gltf.scene;
        scene.add(model);

        // Frame camera to character bounding box
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0.001) {
          const fov = camera.fov * (Math.PI / 180);
          let dist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.4;
          camera.position.set(center.x, center.y + size.y * 0.1, center.z + dist);
          camera.near = Math.max(0.01, maxDim / 100);
          camera.far = Math.max(100, maxDim * 50);
          camera.updateProjectionMatrix();
          controls.target.copy(center);
          controls.maxDistance = maxDim * 10;
          controls.update();

          const existingGrid = scene.children.find(c => c.isGridHelper);
          if (existingGrid) scene.remove(existingGrid);
          const grid = new THREE.GridHelper(maxDim * 3, 20, 0x30363d, 0x161b22);
          grid.position.y = box.min.y;
          scene.add(grid);
        }

        currentGltfClips = gltf.animations || [];
        if (currentGltfClips.length > 0) {
          mixer = new THREE.AnimationMixer(model);

          const animSelect = document.getElementById('animSelect');
          animSelect.innerHTML = '';
          currentGltfClips.forEach((clip, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = (clip.name || `Track ${idx + 1}`) + ` (${clip.duration.toFixed(1)}s)`;
            animSelect.appendChild(opt);
          });
          animSelect.style.display = currentGltfClips.length > 1 ? 'inline-block' : 'none';

          const targetClip = currentGltfClips.find(a => a.name === 'Vid2Anim_Motion') || currentGltfClips[0];
          const targetIdx = currentGltfClips.indexOf(targetClip);
          if (targetIdx >= 0) animSelect.value = targetIdx;

          totalDuration = Math.max(totalDuration, targetClip.duration);
          timeSlider.max = totalDuration;

          currentAction = mixer.clipAction(targetClip);
          currentAction.play();
          currentAction.paused = !isPlaying;

          const boneBadgeText = lastAnimatedBoneCount > 0 ? `${lastAnimatedBoneCount} Animated Bones` : `${targetClip.tracks.length} Tracks`;
          modelMetaBadge.innerText = `${boneBadgeText} • ${targetClip.duration.toFixed(1)}s`;
        } else {
          modelMetaBadge.innerText = 'Static Neutral Pose';
        }

        seekTo(0);
        if (sourceVideo && sourceVideo.src) {
          play();
        }
      });
    }

    function switchAnimation(idx) {
      if (!mixer || !currentGltfClips[idx]) return;
      mixer.stopAllAction();
      currentAction = mixer.clipAction(currentGltfClips[idx]);
      currentAction.play();
      currentAction.paused = !isPlaying;
      totalDuration = Math.max(sourceVideo.duration || 0, currentGltfClips[idx].duration);
      timeSlider.max = totalDuration;
      seekTo(timeSlider.value);
    }

    // Synchronized Playback Functions
    function togglePlayback() {
      if (isPlaying) pause();
      else play();
    }

    function play() {
      isPlaying = true;
      playBtn.className = 'ctrl-btn play-btn active';
      playBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg><span>Pause</span>';

      if (sourceVideo && sourceVideo.src && !sourceVideo.ended) {
        sourceVideo.play().catch(e => console.warn('Video play prevented:', e));
      }
      if (currentAction) {
        currentAction.paused = false;
      }
    }

    function pause() {
      isPlaying = false;
      playBtn.className = 'ctrl-btn play-btn';
      playBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Play</span>';

      if (sourceVideo && sourceVideo.src) {
        sourceVideo.pause();
      }
      if (currentAction) {
        currentAction.paused = true;
      }
    }

    function seekTo(t) {
      const time = Math.max(0, Math.min(totalDuration, parseFloat(t)));
      if (sourceVideo && sourceVideo.src) {
        sourceVideo.currentTime = Math.min(time, sourceVideo.duration || time);
      }
      if (mixer && currentAction) {
        const clipDur = currentAction.getClip().duration;
        const animTime = clipDur > 0 ? (time % clipDur) : 0;
        mixer.setTime(animTime);
      }
      timeSlider.value = time;
      updateTimeDisplays(time);
    }

    function stepFrame(step) {
      pause();
      const dt = step / effectiveFps;
      const cur = (sourceVideo && sourceVideo.src) ? sourceVideo.currentTime : (mixer ? mixer.time : 0);
      seekTo(cur + dt);
    }

    function setSpeed(spd) {
      playbackSpeed = spd;
      if (sourceVideo) sourceVideo.playbackRate = spd;
      if (mixer) mixer.timeScale = spd;
      document.querySelectorAll('.controls-row button').forEach(b => {
        if (b.innerText.endsWith('x')) b.classList.remove('active');
      });
      event.target.classList.add('active');
    }

    function toggleLoop() {
      isLooping = !isLooping;
      const btn = document.getElementById('loopBtn');
      if (isLooping) btn.classList.add('active');
      else btn.classList.remove('active');
    }

    function toggleMute() {
      if (!sourceVideo) return;
      sourceVideo.muted = !sourceVideo.muted;
      const btn = document.getElementById('muteBtn');
      if (sourceVideo.muted) {
        btn.classList.add('active');
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>Muted';
      } else {
        btn.classList.remove('active');
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.08"/></svg>Audio';
      }
    }

    function updateTimeDisplays(cur) {
      const formatTime = (s) => {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        const ms = Math.floor((s % 1) * 100);
        return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
      };
      timecodeText.innerText = `${formatTime(cur)} / ${formatTime(totalDuration)}`;
      const currentFrame = Math.round(cur * effectiveFps);
      const totalFrames = Math.round(totalDuration * effectiveFps);
      framePill.innerText = `Frame ${currentFrame} / ${totalFrames}`;
    }

    // Time Slider scrubbing events
    timeSlider.addEventListener('input', (e) => {
      isSeeking = true;
      seekTo(e.target.value);
    });
    timeSlider.addEventListener('change', (e) => {
      isSeeking = false;
      seekTo(e.target.value);
    });

    function resetCamera() {
      const existingModel = scene.children.find(c => c.isGroup);
      if (existingModel) {
        const box = new THREE.Box3().setFromObject(existingModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let dist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.4;
        camera.position.set(center.x, center.y + size.y * 0.1, center.z + dist);
        controls.target.copy(center);
        controls.update();
      } else {
        camera.position.set(0, 1.4, 3.5);
        controls.target.set(0, 1.0, 0);
        controls.update();
      }
    }

    function setupKeyboardShortcuts() {
      window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.code === 'Space') {
          e.preventDefault();
          togglePlayback();
        } else if (e.code === 'ArrowLeft') {
          e.preventDefault();
          stepFrame(-1);
        } else if (e.code === 'ArrowRight') {
          e.preventDefault();
          stepFrame(1);
        } else if (e.code === 'KeyL') {
          toggleLoop();
        } else if (e.code === 'KeyM') {
          toggleMute();
        }
      });
    }

    // Diagnostics & File Input Handlers
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
        document.getElementById('videoEmptyState').style.display = 'none';

        const fileUrl = URL.createObjectURL(selectedMedia);
        if (selectedMedia.type.startsWith('video/') || selectedMedia.name.match(/\.(mp4|mov|webm)$/i)) {
          sourceImage.style.display = 'none';
          sourceVideo.style.display = 'block';
          sourceVideo.src = fileUrl;
          sourceVideo.load();
        } else {
          sourceVideo.style.display = 'none';
          sourceImage.style.display = 'block';
          sourceImage.src = fileUrl;
          videoMetaBadge.innerText = 'Static Image';
        }
        checkReady();
      }
    });

    document.getElementById('modelInput').addEventListener('change', (e) => {
      if (e.target.files[0]) {
        selectedModel = e.target.files[0];
        document.getElementById('modelName').innerText = selectedModel.name;
        const url = URL.createObjectURL(selectedModel);
        loadGLBModel(url);
        checkReady();
      }
    });

    function selectEngine(engine) {
      document.getElementById('selectedEngine').value = engine;
      document.getElementById('engineCardMediapipe').classList.toggle('active', engine === 'mediapipe');
      document.getElementById('engineCardDwpose').classList.toggle('active', engine === 'dwpose');
    }

    function checkReady() {
      document.getElementById('runBtn').disabled = !(selectedMedia && selectedModel);
    }

    async function startAnimationPipeline() {
      const btn = document.getElementById('runBtn');
      const statusCard = document.getElementById('statusCard');
      const statusTitle = document.getElementById('statusTitle');
      const statusText = document.getElementById('statusText');
      const statusPercent = document.getElementById('statusPercent');
      const progressBar = document.getElementById('progressBar');

      btn.disabled = true;
      statusCard.style.display = 'block';
      statusTitle.innerText = 'Initializing Pipeline';
      statusText.innerText = 'Preparing assets and initializing ML engine...';
      statusPercent.innerText = '0%';
      progressBar.style.width = '0%';

      const formData = new FormData();
      formData.append('media', selectedMedia);
      formData.append('model', selectedModel);
      formData.append('device', document.getElementById('deviceSelect').value);
      formData.append('engine', document.getElementById('selectedEngine').value);

      // Real-time progress poller
      const progressTimer = setInterval(async () => {
        try {
          const res = await fetch('/api/progress');
          if (res.ok) {
            const data = await res.json();
            if (data.active && data.percent > 0) {
              progressBar.style.width = data.percent + '%';
              statusPercent.innerText = data.percent + '%';
              statusTitle.innerText = data.step || 'Processing';
              statusText.innerText = data.message || 'Extracting motion...';
            }
          }
        } catch (e) {}
      }, 200);

      try {
        const response = await fetch('/api/animate', {
          method: 'POST',
          body: formData,
        });

        clearInterval(progressTimer);

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.error || 'Animation pipeline failed');
        }

        statusPercent.innerText = '95%';
        progressBar.style.width = '95%';
        statusText.innerText = 'Packing and loading animated character...';

        const boneCountHeader = response.headers.get('X-Animated-Bones');
        if (boneCountHeader) {
          lastAnimatedBoneCount = parseInt(boneCountHeader, 10);
        }

        const blob = await response.blob();
        lastResultBlobUrl = URL.createObjectURL(blob);
        loadGLBModel(lastResultBlobUrl);

        statusPercent.innerText = '100%';
        progressBar.style.width = '100%';
        statusTitle.innerText = 'Complete';
        const boneNotice = lastAnimatedBoneCount > 0 ? ` (${lastAnimatedBoneCount} bones animated: spine, limbs, twists, fingers)` : '';
        statusText.innerText = `Synced side-by-side playback ready!${boneNotice}`;
        document.getElementById('downloadBtn').style.display = 'flex';
      } catch (err) {
        clearInterval(progressTimer);
        statusTitle.innerText = 'Pipeline Error';
        statusText.innerText = err.message;
        statusPercent.innerText = '0%';
        progressBar.style.width = '0%';
      } finally {
        clearInterval(progressTimer);
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

        elif parsed_path.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(PIPELINE_PROGRESS).encode("utf-8"))

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
            engine_str = fields.get("engine", "mediapipe")

            def on_progress(step: str, progress: float, message: str) -> None:
                if step == "EXTRACT":
                    pct = 10 + int(progress * 55)
                elif step == "RETARGET":
                    pct = 70 + int(progress * 15)
                elif step == "EXPORT":
                    pct = 85 + int(progress * 10)
                elif step == "PIPELINE":
                    pct = int(progress * 100)
                else:
                    pct = int(progress * 100)
                pct = max(1, min(98, pct))
                PIPELINE_PROGRESS["active"] = True
                PIPELINE_PROGRESS["percent"] = max(PIPELINE_PROGRESS.get("percent", 0), pct)
                PIPELINE_PROGRESS["step"] = step
                PIPELINE_PROGRESS["message"] = message

            PIPELINE_PROGRESS["active"] = True
            PIPELINE_PROGRESS["percent"] = 5
            PIPELINE_PROGRESS["step"] = "EXTRACT"
            PIPELINE_PROGRESS["message"] = f"Initializing tracking pipeline ({engine_str.upper()})..."

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
                    engine=engine_str,
                    max_duration_sec=30.0,
                    smooth=True,
                )
                reporter = ProgressReporter(on_report=on_progress)
                pipeline = Vid2AnimPipeline(config, reporter=reporter)
                result = pipeline.execute()

                if not result.success:
                    PIPELINE_PROGRESS["active"] = False
                    PIPELINE_PROGRESS["percent"] = 0
                    PIPELINE_PROGRESS["step"] = "Error"
                    PIPELINE_PROGRESS["message"] = result.error_message or "Execution failed"
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": result.error_message or "Execution failed"}).encode("utf-8"))
                    return

                # Send back the generated GLB binary
                glb_data = output_glb.read_bytes()
                PIPELINE_PROGRESS["active"] = False
                PIPELINE_PROGRESS["percent"] = 100
                PIPELINE_PROGRESS["step"] = "Complete"
                PIPELINE_PROGRESS["message"] = f"Complete: {result.mapped_bones_count} bones animated."

                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(glb_data)))
                self.send_header("X-Animated-Bones", str(result.mapped_bones_count))
                self.send_header("Access-Control-Expose-Headers", "X-Animated-Bones")
                self.end_headers()
                self.wfile.write(glb_data)

        except Exception as e:
            PIPELINE_PROGRESS["active"] = False
            PIPELINE_PROGRESS["percent"] = 0
            PIPELINE_PROGRESS["step"] = "Error"
            PIPELINE_PROGRESS["message"] = str(e)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the built-in local web server."""
    with ThreadedTCPServer((host, port), Vid2AnimHTTPHandler) as httpd:
        print("=" * 60)
        print(f" vid2anim Web UI & API Server running at:")
        print(f"   http://{host}:{port}")
        print(" Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
