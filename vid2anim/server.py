"""Zero-Dependency Web UI & API Server for vid2anim.

Provides a modern browser interface with drag-and-drop file upload, real-time
progress updates, dual-AI model selector, and an interactive Three.js 3D viewport.
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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
  <style>
    :root {
      --bg-dark: #07090e;
      --bg-panel: #0b0f19;
      --bg-panel-trans: rgba(11, 15, 25, 0.85);
      --bg-card: #121829;
      --bg-card-hover: #182138;
      --border: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(99, 102, 241, 0.4);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
      --cyan: #06b6d4;
      --cyan-glow: rgba(6, 182, 212, 0.3);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --amber: #f59e0b;
      --text: #cbd5e1;
      --text-bright: #ffffff;
      --text-dim: #64748b;
      --shadow-card: 0 8px 30px rgba(0, 0, 0, 0.4);
      --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.25);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    body {
      font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 85% 90%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      user-select: none;
    }

    /* Top Navigation Header */
    header {
      padding: 10px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-panel-trans);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: var(--accent-grad);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-pill {
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 999px;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.35);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .quick-demo-btn {
      padding: 5px 12px;
      border-radius: 8px;
      font-size: 0.78rem;
      font-weight: 600;
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.35);
      color: #a5b4fc;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .quick-demo-btn:hover {
      background: rgba(99, 102, 241, 0.25);
      color: #ffffff;
      box-shadow: 0 0 14px var(--accent-glow);
      transform: translateY(-1px);
    }
    .badge {
      font-size: 0.74rem;
      padding: 5px 11px;
      border-radius: 999px;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      backdrop-filter: blur(8px);
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

    /* Main Container */
    .main-container {
      display: grid;
      grid-template-columns: 370px 1fr;
      flex: 1;
      overflow: hidden;
    }

    /* Sidebar Studio Panel */
    .sidebar {
      padding: 18px 20px 24px;
      border-right: 1px solid var(--border);
      background: var(--bg-panel-trans);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      z-index: 10;
    }

    .section-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: var(--shadow-card);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .section-card:hover {
      border-color: rgba(255, 255, 255, 0.14);
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .section-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-bright);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-number {
      font-size: 0.7rem;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .section-tag {
      font-size: 0.7rem;
      font-weight: 500;
      color: var(--text-dim);
    }

    /* Upload Zones */
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
      background: rgba(99, 102, 241, 0.08);
      transform: translateY(-1px);
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }
    .upload-zone svg {
      stroke: #818cf8;
      margin-bottom: 6px;
      filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.3));
    }
    .upload-zone p { font-size: 0.82rem; color: var(--text); font-weight: 500; }
    .upload-zone span { font-size: 0.72rem; color: var(--text-dim); }

    .file-pill {
      display: none;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-radius: 8px;
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.3);
      font-size: 0.78rem;
      color: #cbd5e1;
      margin-top: 4px;
    }
    .file-pill-info {
      display: flex;
      align-items: center;
      gap: 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .file-pill-name {
      font-weight: 600;
      color: #ffffff;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .file-pill-btn {
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.08);
      border: none;
      color: #94a3b8;
      font-size: 0.7rem;
      cursor: pointer;
    }
    .file-pill-btn:hover { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    .sample-pill-btn {
      width: 100%;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      color: #94a3b8;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .sample-pill-btn:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      color: #ffffff;
    }

    /* Dual-Engine AI Cards */
    .engine-cards {
      display: flex;
      flex-direction: column;
      gap: 9px;
    }
    .engine-card {
      padding: 11px 13px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.02);
      border: 1.5px solid var(--border);
      cursor: pointer;
      transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
    }
    .engine-card:hover {
      border-color: rgba(99, 102, 241, 0.45);
      background: var(--bg-card-hover);
      transform: translateY(-1px);
    }
    .engine-card.active {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.12);
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.25);
    }
    .engine-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 5px;
    }
    .engine-title {
      font-size: 0.86rem;
      font-weight: 700;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .engine-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--text-dim);
      transition: all 0.2s;
    }
    .engine-card.active .engine-dot {
      background: #818cf8;
      box-shadow: 0 0 8px #6366f1;
    }
    .engine-tag {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .tag-fast {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .tag-precision {
      background: rgba(139, 92, 246, 0.18);
      color: #c084fc;
      border: 1px solid rgba(139, 92, 246, 0.45);
    }
    .engine-desc {
      font-size: 0.73rem;
      color: var(--text-dim);
      line-height: 1.4;
    }

    /* Settings & Controls */
    select, button {
      font-family: inherit;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      color: var(--text-bright);
      font-size: 0.84rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    select:focus, button:focus {
      outline: none;
      border-color: var(--accent);
    }

    /* Primary Action Button */
    button.primary {
      background: var(--accent-grad);
      border: none;
      font-weight: 700;
      font-size: 0.94rem;
      padding: 13px;
      border-radius: 10px;
      box-shadow: 0 4px 20px var(--accent-glow);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      position: relative;
      overflow: hidden;
      letter-spacing: 0.2px;
    }
    button.primary:hover:not(:disabled) {
      filter: brightness(1.1);
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45);
    }
    button:disabled { opacity: 0.35; cursor: not-allowed; box-shadow: none; }

    /* Interactive Stepper Pipeline Status */
    .status-card {
      padding: 14px;
      border-radius: 10px;
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
      font-size: 0.82rem;
      box-shadow: var(--shadow-glow);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      color: var(--text-bright);
    }
    .pipeline-stepper {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 4px;
      margin: 4px 0;
    }
    .step-pill {
      font-size: 0.65rem;
      font-weight: 600;
      padding: 4px 2px;
      text-align: center;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-dim);
      border: 1px solid rgba(255, 255, 255, 0.05);
      transition: all 0.25s;
    }
    .step-pill.active {
      background: rgba(99, 102, 241, 0.25);
      color: #a5b4fc;
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
    }
    .step-pill.done {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.35);
    }
    .progress-bar-bg {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--accent-grad);
      width: 0%;
      transition: width 0.3s ease;
      box-shadow: 0 0 10px var(--accent);
    }
    .status-detail {
      font-size: 0.74rem;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .download-btn {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.45);
      color: #34d399;
      font-weight: 700;
      font-size: 0.88rem;
      padding: 12px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
      transition: all 0.2s;
    }
    .download-btn:hover {
      background: rgba(16, 185, 129, 0.25);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
    }

    /* Stage Area */
    .stage-area {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      background: #04060a;
      position: relative;
    }
    .stage-header {
      padding: 8px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-panel-trans);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
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
      font-weight: 600;
      border: none;
      background: transparent;
      color: var(--text-dim);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .view-tab.active {
      background: var(--accent);
      color: #ffffff;
      box-shadow: 0 2px 10px var(--accent-glow);
    }
    .stage-tools {
      display: flex;
      align-items: center;
      gap: 6px;
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
      background: #030509;
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
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(8px);
      padding: 5px 11px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.72rem;
      font-weight: 700;
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

    /* Video player */
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

    /* 3D Viewport */
    #modelViewport {
      background: radial-gradient(circle at center, #0e1629 0%, #040711 100%);
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
      color: rgba(255, 255, 255, 0.4);
      background: rgba(7, 9, 14, 0.75);
      backdrop-filter: blur(4px);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.06);
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
      stroke: rgba(255, 255, 255, 0.16);
      margin-bottom: 12px;
    }
    .viewport-empty h4 {
      font-size: 0.98rem;
      color: var(--text-bright);
      margin-bottom: 4px;
    }
    .viewport-empty p {
      font-size: 0.78rem;
      max-width: 240px;
      line-height: 1.4;
    }

    /* Floating Viewport HUD buttons */
    .viewport-hud {
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 5;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(11, 15, 25, 0.8);
      backdrop-filter: blur(8px);
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .hud-btn {
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-dim);
      display: inline-flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .hud-btn:hover {
      color: #ffffff;
      background: rgba(255, 255, 255, 0.05);
    }
    .hud-btn.active {
      color: #a5b4fc;
      background: rgba(99, 102, 241, 0.2);
      border-color: rgba(99, 102, 241, 0.4);
    }

    /* Bottom Docked Playback Bar */
    .sync-controller {
      background: var(--bg-panel-trans);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-top: 1px solid var(--border);
      padding: 10px 24px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 10;
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
      width: 15px;
      height: 15px;
      border-radius: 50%;
      background: #818cf8;
      box-shadow: 0 0 12px #6366f1;
      cursor: pointer;
      border: 2px solid #ffffff;
      transition: transform 0.15s;
    }
    .time-slider::-webkit-slider-thumb:hover {
      transform: scale(1.3);
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
      font-weight: 600;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-bright);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .ctrl-btn:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.15);
      transform: translateY(-1px);
    }
    .ctrl-btn.play-btn {
      background: var(--accent);
      border-color: var(--accent);
      padding: 6px 18px;
      font-weight: 700;
      box-shadow: 0 0 12px var(--accent-glow);
    }
    .ctrl-btn.play-btn:hover {
      background: var(--accent-hover);
      box-shadow: 0 0 18px var(--accent-glow);
    }
    .ctrl-btn.active {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent);
      color: #a5b4fc;
    }
    .ctrl-key {
      font-size: 0.65rem;
      padding: 1px 4px;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.08);
      color: #94a3b8;
      margin-left: 2px;
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
      padding: 3px 7px;
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
      <div class="brand-pill">Studio Pro v2.5</div>
    </div>
    <div class="header-actions">
      <button class="quick-demo-btn" onclick="loadQuickDemo()" title="Instantly load sample video and 3D character for 1-click test">
        <span>⚡</span> Quick Demo Preset
      </button>
      <div id="syncBadge" class="badge badge-sync active">
        <span class="badge-dot"></span>
        <span id="syncText">Dual-Viewport Synced</span>
      </div>
      <div id="deviceBadge" class="badge badge-device">Loading device...</div>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <!-- Section 1: Media Upload -->
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">
            <span class="section-number">01</span> Source Footage
          </div>
          <span class="section-tag">Max 30s</span>
        </div>
        <div class="upload-zone" id="mediaZone" onclick="document.getElementById('mediaInput').click()">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          <p>Drop Video or Photo</p>
          <span>MP4, MOV, WEBM, JPG, PNG</span>
        </div>
        <div id="mediaPill" class="file-pill">
          <div class="file-pill-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            <span id="mediaName" class="file-pill-name"></span>
          </div>
          <button class="file-pill-btn" onclick="clearMedia(event)">Clear</button>
        </div>
        <button type="button" class="sample-pill-btn" onclick="loadSampleVideo()">
          <span>⚡</span> Use "Phone Hold" Sample Clip
        </button>
        <input type="file" id="mediaInput" accept="video/mp4,video/quicktime,video/webm,image/jpeg,image/png" style="display:none">
      </div>

      <!-- Section 2: 3D Model Upload -->
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">
            <span class="section-number">02</span> Target 3D Character
          </div>
          <span class="section-tag">Rigged Mesh</span>
        </div>
        <div class="upload-zone" id="modelZone" onclick="document.getElementById('modelInput').click()">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
          <p>Drop 3D Character Model</p>
          <span>GLB, GLTF, FBX, OBJ</span>
        </div>
        <div id="modelPill" class="file-pill">
          <div class="file-pill-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            <span id="modelName" class="file-pill-name"></span>
          </div>
          <button class="file-pill-btn" onclick="clearModel(event)">Clear</button>
        </div>
        <button type="button" class="sample-pill-btn" onclick="loadSampleModel()">
          <span>⚡</span> Use "Cyborg Oni" 3D Character
        </button>
        <input type="file" id="modelInput" accept=".glb,.gltf,.fbx,.obj" style="display:none">
      </div>

      <!-- Section 3: Dual-Engine Model Selector -->
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">
            <span class="section-number">03</span> Tracking AI Model
          </div>
          <span class="section-tag">Dual-Engine</span>
        </div>
        <div class="engine-cards">
          <div class="engine-card active" id="engineCardMediapipe" onclick="selectEngine('mediapipe')">
            <div class="engine-card-header">
              <div class="engine-title">
                <span class="engine-dot"></span>
                MediaPipe Pose
              </div>
              <span class="engine-tag tag-fast">Fast CPU (~30 FPS)</span>
            </div>
            <p class="engine-desc">Lightweight real-time tracking. Instant startup, low memory, robust full-body kinematics.</p>
          </div>
          <div class="engine-card" id="engineCardDwpose" onclick="selectEngine('dwpose')">
            <div class="engine-card-header">
              <div class="engine-title">
                <span class="engine-dot"></span>
                DWPose WholeBody
              </div>
              <span class="engine-tag tag-precision">133 Pts Deep AI</span>
            </div>
            <p class="engine-desc">133 whole-body landmarks + full 21-joint per-hand finger curling & articulation via ONNX.</p>
          </div>
        </div>
        <input type="hidden" id="selectedEngine" value="mediapipe">
      </div>

      <!-- Section 4: Motion Settings -->
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">
            <span class="section-number">04</span> Engine & Tuning
          </div>
          <span class="section-tag">Anti-Jitter</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-dim);">
            <span>Compute Device</span>
            <span style="color:var(--cyan)">Auto-Optimized</span>
          </div>
          <select id="deviceSelect">
            <option value="cpu" selected>CPU (Lightweight / Stable)</option>
            <option value="cuda">GPU (NVIDIA CUDA Acceleration)</option>
          </select>
        </div>
      </div>

      <!-- Section 5: Primary Action & Status -->
      <button id="runBtn" class="primary" onclick="startAnimationPipeline()" disabled>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Animate 3D Character
      </button>

      <!-- Live Pipeline Stepper Status -->
      <div id="statusCard" class="status-card" style="display:none">
        <div class="status-header">
          <span id="statusTitle">Processing</span>
          <span id="statusPercent" style="color:var(--cyan)">0%</span>
        </div>
        <div class="pipeline-stepper">
          <div id="stepPill1" class="step-pill">1. Ingest</div>
          <div id="stepPill2" class="step-pill">2. Extract</div>
          <div id="stepPill3" class="step-pill">3. Retarget</div>
          <div id="stepPill4" class="step-pill">4. Pack GLB</div>
        </div>
        <div class="progress-bar-bg">
          <div id="progressBar" class="progress-bar-fill"></div>
        </div>
        <div class="status-detail">
          <span id="statusText">Preparing assets...</span>
          <span id="statusTimer" style="font-family:'JetBrains Mono'">00:00.0s</span>
        </div>
      </div>

      <button id="downloadBtn" class="download-btn" onclick="downloadResult()" style="display:none">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
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
            3D Solo
          </button>
          <button class="view-tab" id="tabVideo" onclick="setViewMode('video')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Video Solo
          </button>
        </div>

        <div class="stage-tools">
          <select id="animSelect" onchange="switchAnimation(this.value)" style="display:none; font-size:0.75rem; padding:4px 8px;"></select>
          <button class="ctrl-btn" onclick="resetCamera()" title="Reset 3D Camera [R]">
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
            <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
            <h4>Reference Footage</h4>
            <p>Upload a video on the left or click "Quick Demo" to preview here</p>
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

          <!-- Floating Viewport HUD -->
          <div class="viewport-hud">
            <button class="hud-btn" id="btnSkeleton" onclick="toggleSkeleton()" title="Toggle Skeleton Overlay [B]">
              🦴 Bones
            </button>
            <button class="hud-btn active" id="btnGrid" onclick="toggleGrid()" title="Toggle Floor Grid [G]">
              📐 Grid
            </button>
            <button class="hud-btn" id="btnWireframe" onclick="toggleWireframe()" title="Toggle Wireframe">
              🌐 Wire
            </button>
          </div>

          <div id="modelEmptyState" class="viewport-empty">
            <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            <h4>3D Character Viewport</h4>
            <p>Upload character on the left or use the "Quick Demo" button</p>
          </div>
          <div id="viewport3d"></div>
          <div class="viewport-hints">Left Click: Rotate | Right Click: Pan | Scroll: Zoom | Space: Play/Pause</div>
        </div>
      </div>

      <!-- Bottom Docked Playback Bar -->
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
              <span class="ctrl-key">Space</span>
            </button>
            <button class="ctrl-btn" onclick="stepFrame(-1)" title="Previous Frame [Left Arrow]">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
              -1F <span class="ctrl-key">←</span>
            </button>
            <button class="ctrl-btn" onclick="stepFrame(1)" title="Next Frame [Right Arrow]">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
              +1F <span class="ctrl-key">→</span>
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
            <button id="loopBtn" class="ctrl-btn active" onclick="toggleLoop()" title="Toggle Looping [L]">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
              Loop
            </button>
            <button id="muteBtn" class="ctrl-btn active" onclick="toggleMute()" title="Toggle Audio [M]">
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
    let skeletonHelper = null;
    let isSkeletonVisible = false;
    let isGridVisible = true;
    let isWireframe = false;

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
      renderer.shadowMap.enabled = true;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0, 1.0, 0);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.update();

      const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
      scene.add(hemiLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
      dirLight.position.set(2, 4, 3);
      scene.add(dirLight);

      const backLight = new THREE.DirectionalLight(0x6366f1, 0.8);
      backLight.position.set(-2, 2, -2);
      scene.add(backLight);

      const grid = new THREE.GridHelper(10, 20, 0x4f46e5, 0x161b22);
      grid.name = 'floorGrid';
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

    // Main Render Loop
    function animate() {
      requestAnimationFrame(animate);

      if (isPlaying && !isSeeking) {
        let curTime = 0;

        // If source video exists, lockstep synchronizer
        if (sourceVideo && sourceVideo.src && !sourceVideo.paused && sourceVideo.readyState >= 2) {
          curTime = sourceVideo.currentTime;
          if (mixer && currentAction) {
            const clipDur = currentAction.getClip().duration;
            const animTime = clipDur > 0 ? (curTime % clipDur) : 0;
            mixer.setTime(animTime);
          }
        } else if (mixer && currentAction && !currentAction.paused) {
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

      setTimeout(onWindowResize, 40);
    }

    // GLB Loading
    function loadGLBModel(url) {
      document.getElementById('modelEmptyState').style.display = 'none';

      const loader = new THREE.GLTFLoader();
      loader.load(url, (gltf) => {
        if (mixer) mixer.stopAllAction();
        scene.children.filter(c => c.isGroup || c.isMesh || c.isSkeletonHelper).forEach(c => scene.remove(c));

        const model = gltf.scene;
        scene.add(model);

        // Skeleton overlay
        skeletonHelper = new THREE.SkeletonHelper(model);
        skeletonHelper.visible = isSkeletonVisible;
        skeletonHelper.material.linewidth = 2;
        scene.add(skeletonHelper);

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

          const existingGrid = scene.children.find(c => c.name === 'floorGrid');
          if (existingGrid) scene.remove(existingGrid);
          const grid = new THREE.GridHelper(maxDim * 3, 20, 0x4f46e5, 0x161b22);
          grid.name = 'floorGrid';
          grid.position.y = box.min.y;
          grid.visible = isGridVisible;
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

          const boneBadgeText = lastAnimatedBoneCount > 0 ? `${lastAnimatedBoneCount} Bones Animated` : `${targetClip.tracks.length} Tracks`;
          modelMetaBadge.innerText = `${boneBadgeText} • ${targetClip.duration.toFixed(1)}s`;
        } else {
          modelMetaBadge.innerText = 'Static Neutral Mesh';
        }

        seekTo(0);
        if (sourceVideo && sourceVideo.src) {
          play();
        }
      });
    }

    function switchAnimation(clipIndex) {
      if (!mixer || !currentGltfClips[clipIndex]) return;
      if (currentAction) currentAction.stop();
      currentAction = mixer.clipAction(currentGltfClips[clipIndex]);
      currentAction.play();
      currentAction.paused = !isPlaying;
      totalDuration = currentGltfClips[clipIndex].duration;
      timeSlider.max = totalDuration;
      seekTo(0);
    }

    // Viewport HUD Toggles
    function toggleSkeleton() {
      isSkeletonVisible = !isSkeletonVisible;
      if (skeletonHelper) skeletonHelper.visible = isSkeletonVisible;
      document.getElementById('btnSkeleton').classList.toggle('active', isSkeletonVisible);
    }

    function toggleGrid() {
      isGridVisible = !isGridVisible;
      const grid = scene.children.find(c => c.name === 'floorGrid');
      if (grid) grid.visible = isGridVisible;
      document.getElementById('btnGrid').classList.toggle('active', isGridVisible);
    }

    function toggleWireframe() {
      isWireframe = !isWireframe;
      scene.traverse((node) => {
        if (node.isMesh && node.material) {
          node.material.wireframe = isWireframe;
        }
      });
      document.getElementById('btnWireframe').classList.toggle('active', isWireframe);
    }

    // Playback Controls
    function togglePlayback() {
      if (isPlaying) pause();
      else play();
    }

    function play() {
      isPlaying = true;
      playBtn.classList.add('active');
      playBtn.querySelector('span').innerText = 'Pause';
      if (sourceVideo && sourceVideo.src) sourceVideo.play();
      if (currentAction) currentAction.paused = false;
      clock.start();
    }

    function pause() {
      isPlaying = false;
      playBtn.classList.remove('active');
      playBtn.querySelector('span').innerText = 'Play';
      if (sourceVideo && sourceVideo.src) sourceVideo.pause();
      if (currentAction) currentAction.paused = true;
      clock.stop();
    }

    function seekTo(time) {
      const t = parseFloat(time);
      if (sourceVideo && sourceVideo.src) sourceVideo.currentTime = t;
      if (mixer && currentAction) {
        const clipDur = currentAction.getClip().duration;
        const animTime = clipDur > 0 ? (t % clipDur) : 0;
        mixer.setTime(animTime);
      }
      updateTimeDisplays(t);
    }

    function stepFrame(delta) {
      pause();
      const stepSec = 1.0 / effectiveFps;
      const newTime = Math.max(0, Math.min(totalDuration, (parseFloat(timeSlider.value) || 0) + delta * stepSec));
      timeSlider.value = newTime;
      seekTo(newTime);
    }

    function setSpeed(spd) {
      playbackSpeed = spd;
      document.querySelectorAll('#syncController .ctrl-group button').forEach(b => {
        if (b.innerText.endsWith('x')) b.classList.remove('active');
      });
      if (sourceVideo && sourceVideo.src) sourceVideo.playbackRate = spd;
      event.target.classList.add('active');
    }

    function toggleLoop() {
      isLooping = !isLooping;
      document.getElementById('loopBtn').classList.toggle('active', isLooping);
      if (sourceVideo) sourceVideo.loop = isLooping;
    }

    function toggleMute() {
      if (sourceVideo) {
        sourceVideo.muted = !sourceVideo.muted;
        document.getElementById('muteBtn').classList.toggle('active', sourceVideo.muted);
        document.getElementById('muteBtn').innerText = sourceVideo.muted ? 'Muted' : 'Unmuted';
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

    // Time Slider scrubbing
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
        } else if (e.code === 'KeyR') {
          resetCamera();
        } else if (e.code === 'KeyB') {
          toggleSkeleton();
        } else if (e.code === 'KeyG') {
          toggleGrid();
        }
      });
    }

    // Diagnostics & File Handlers
    fetch('/api/diagnostics')
      .then(r => r.json())
      .then(d => {
        const badge = document.getElementById('deviceBadge');
        badge.innerText = d.cuda_available ? 'GPU Mode (CUDA Available)' : 'CPU Mode (Optimized)';
      });

    document.getElementById('mediaInput').addEventListener('change', (e) => {
      if (e.target.files[0]) {
        handleMediaFile(e.target.files[0]);
      }
    });

    document.getElementById('modelInput').addEventListener('change', (e) => {
      if (e.target.files[0]) {
        handleModelFile(e.target.files[0]);
      }
    });

    function handleMediaFile(file) {
      selectedMedia = file;
      document.getElementById('mediaName').innerText = file.name;
      document.getElementById('mediaPill').style.display = 'flex';
      document.getElementById('mediaZone').style.display = 'none';
      document.getElementById('videoEmptyState').style.display = 'none';

      const fileUrl = URL.createObjectURL(file);
      if (file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|webm)$/i)) {
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

    function clearMedia(e) {
      if (e) e.stopPropagation();
      selectedMedia = null;
      document.getElementById('mediaInput').value = '';
      document.getElementById('mediaPill').style.display = 'none';
      document.getElementById('mediaZone').style.display = 'block';
      document.getElementById('videoEmptyState').style.display = 'flex';
      sourceVideo.style.display = 'none';
      sourceVideo.src = '';
      sourceImage.style.display = 'none';
      sourceImage.src = '';
      videoMetaBadge.innerText = 'No media';
      checkReady();
    }

    function handleModelFile(file) {
      selectedModel = file;
      document.getElementById('modelName').innerText = file.name;
      document.getElementById('modelPill').style.display = 'flex';
      document.getElementById('modelZone').style.display = 'none';
      const url = URL.createObjectURL(file);
      loadGLBModel(url);
      checkReady();
    }

    function clearModel(e) {
      if (e) e.stopPropagation();
      selectedModel = null;
      document.getElementById('modelInput').value = '';
      document.getElementById('modelPill').style.display = 'none';
      document.getElementById('modelZone').style.display = 'block';
      document.getElementById('modelEmptyState').style.display = 'flex';
      if (mixer) mixer.stopAllAction();
      scene.children.filter(c => c.isGroup || c.isMesh || c.isSkeletonHelper).forEach(c => scene.remove(c));
      modelMetaBadge.innerText = 'Waiting for model';
      checkReady();
    }

    // One-click Sample Loaders
    async function loadSampleVideo() {
      try {
        const res = await fetch('/api/samples/video');
        if (!res.ok) throw new Error('Sample video not found on server');
        const blob = await res.blob();
        const file = new File([blob], '3 phone hold.mp4', { type: 'video/mp4' });
        handleMediaFile(file);
      } catch (e) {
        alert('Could not load sample video: ' + e.message);
      }
    }

    async function loadSampleModel() {
      try {
        const res = await fetch('/api/samples/model');
        if (!res.ok) throw new Error('Sample model not found on server');
        const blob = await res.blob();
        const file = new File([blob], 'cyborg_oni.glb', { type: 'model/gltf-binary' });
        handleModelFile(file);
      } catch (e) {
        alert('Could not load sample model: ' + e.message);
      }
    }

    async function loadQuickDemo() {
      await loadSampleVideo();
      await loadSampleModel();
      selectEngine('dwpose');
    }

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
      const statusTimer = document.getElementById('statusTimer');
      const progressBar = document.getElementById('progressBar');

      const pill1 = document.getElementById('stepPill1');
      const pill2 = document.getElementById('stepPill2');
      const pill3 = document.getElementById('stepPill3');
      const pill4 = document.getElementById('stepPill4');

      btn.disabled = true;
      statusCard.style.display = 'flex';
      statusTitle.innerText = 'Pipeline Running';
      statusText.innerText = 'Preparing assets and initializing ML engine...';
      statusPercent.innerText = '0%';
      progressBar.style.width = '0%';

      [pill1, pill2, pill3, pill4].forEach(p => p.className = 'step-pill');
      pill1.classList.add('active');

      const startTime = performance.now();
      const timerInterval = setInterval(() => {
        const elapsed = (performance.now() - startTime) / 1000;
        statusTimer.innerText = `${elapsed.toFixed(1)}s`;
      }, 100);

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
              statusText.innerText = data.message || 'Extracting motion...';

              if (data.percent < 15) {
                pill1.className = 'step-pill active';
              } else if (data.percent < 70) {
                pill1.className = 'step-pill done';
                pill2.className = 'step-pill active';
              } else if (data.percent < 85) {
                pill1.className = 'step-pill done';
                pill2.className = 'step-pill done';
                pill3.className = 'step-pill active';
              } else {
                pill1.className = 'step-pill done';
                pill2.className = 'step-pill done';
                pill3.className = 'step-pill done';
                pill4.className = 'step-pill active';
              }
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
        clearInterval(timerInterval);

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.error || 'Animation pipeline failed');
        }

        [pill1, pill2, pill3, pill4].forEach(p => p.className = 'step-pill done');
        statusPercent.innerText = '95%';
        progressBar.style.width = '95%';
        statusText.innerText = 'Packing animated character...';

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
        const totalSec = ((performance.now() - startTime) / 1000).toFixed(1);
        statusTimer.innerText = `${totalSec}s`;
        const boneNotice = lastAnimatedBoneCount > 0 ? ` (${lastAnimatedBoneCount} bones animated: spine, limbs, fingers)` : '';
        statusText.innerText = `Synchronized playback ready!${boneNotice}`;
        document.getElementById('downloadBtn').style.display = 'flex';
      } catch (err) {
        clearInterval(progressTimer);
        clearInterval(timerInterval);
        statusTitle.innerText = 'Pipeline Error';
        statusText.innerText = err.message;
        statusPercent.innerText = '0%';
        progressBar.style.width = '0%';
      } finally {
        clearInterval(progressTimer);
        clearInterval(timerInterval);
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

        elif parsed_path.path == "/api/samples/video":
            video_path = Path("vid/3 phone hold.mp4")
            if video_path.exists():
                data = video_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed_path.path == "/api/samples/model":
            model_path = Path("glb/cyborg_oni.glb")
            if model_path.exists():
                data = model_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

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
