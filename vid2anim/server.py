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
  <title>vid2anim</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
  <style>
    :root {
      --bg-dark: #0a0c10;
      --bg-panel: #11141a;
      --bg-card: #161a22;
      --bg-card-hover: #1c212b;
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.16);
      --accent: #4f46e5;
      --accent-hover: #4338ca;
      --emerald: #10b981;
      --text: #cbd5e1;
      --text-bright: #f8fafc;
      --text-dim: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-dark);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      user-select: none;
    }

    /* Top Navigation Header */
    header {
      height: 48px;
      padding: 0 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-panel);
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-logo {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-bright);
      letter-spacing: -0.3px;
    }
    .brand-logo svg {
      color: var(--accent);
    }
    .version-tag {
      font-size: 0.68rem;
      font-weight: 600;
      color: var(--text-dim);
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .device-pill {
      font-size: 0.7rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.25);
      font-family: 'JetBrains Mono', monospace;
    }
    .header-link {
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 0.74rem;
      font-weight: 500;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: all 0.15s;
    }
    .header-link:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-bright);
      border-color: var(--border-hover);
    }

    /* Main Container */
    .main-container {
      display: grid;
      grid-template-columns: 310px 1fr;
      flex: 1;
      overflow: hidden;
    }

    /* Sidebar Tools Panel */
    .sidebar {
      padding: 14px;
      border-right: 1px solid var(--border);
      background: var(--bg-panel);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 10;
    }

    .section-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .section-title {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    /* Upload Dropzones */
    .upload-zone {
      border: 1px dashed var(--border);
      border-radius: 6px;
      padding: 14px 10px;
      text-align: center;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.015);
      transition: all 0.15s;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(79, 70, 229, 0.05);
    }
    .upload-zone svg {
      stroke: var(--text-dim);
      margin-bottom: 4px;
    }
    .upload-zone:hover svg {
      stroke: #818cf8;
    }
    .upload-zone p { font-size: 0.78rem; color: var(--text); font-weight: 500; }
    .upload-zone span { font-size: 0.68rem; color: var(--text-dim); }

    .file-pill {
      display: none;
      align-items: center;
      justify-content: space-between;
      padding: 7px 10px;
      border-radius: 6px;
      background: rgba(79, 70, 229, 0.08);
      border: 1px solid rgba(79, 70, 229, 0.25);
      font-size: 0.74rem;
    }
    .file-pill-info {
      display: flex;
      align-items: center;
      gap: 6px;
      overflow: hidden;
      white-space: nowrap;
    }
    .file-pill-name {
      font-weight: 500;
      color: var(--text-bright);
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .file-pill-btn {
      padding: 2px 6px;
      border-radius: 4px;
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-dim);
      font-size: 0.68rem;
      cursor: pointer;
    }
    .file-pill-btn:hover { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }

    /* Segmented Control for Engine */
    .segmented-control {
      display: grid;
      grid-template-columns: 1fr 1fr;
      background: #0d1016;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }
    .segment-btn {
      padding: 7px 8px;
      border-radius: 4px;
      font-size: 0.74rem;
      font-weight: 500;
      background: transparent;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }
    .segment-btn:hover {
      color: var(--text-bright);
    }
    .segment-btn.active {
      background: var(--bg-card);
      color: var(--text-bright);
      font-weight: 600;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Form Controls */
    select, button {
      font-family: inherit;
    }
    select {
      width: 100%;
      padding: 7px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #0d1016;
      color: var(--text-bright);
      font-size: 0.78rem;
      cursor: pointer;
      outline: none;
    }
    select:focus {
      border-color: var(--accent);
    }

    /* Action Buttons */
    button.primary {
      background: var(--accent);
      border: none;
      color: #ffffff;
      font-weight: 600;
      font-size: 0.84rem;
      padding: 10px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      transition: background 0.15s;
    }
    button.primary:hover:not(:disabled) {
      background: var(--accent-hover);
    }
    button:disabled { opacity: 0.35; cursor: not-allowed; }

    /* Status & Progress */
    .status-card {
      padding: 10px 12px;
      border-radius: 6px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      font-size: 0.78rem;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 600;
      color: var(--text-bright);
      font-size: 0.76rem;
    }
    .pipeline-stepper {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 3px;
    }
    .step-pill {
      font-size: 0.65rem;
      font-weight: 500;
      padding: 3px 2px;
      text-align: center;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-dim);
      border: 1px solid var(--border);
    }
    .step-pill.active {
      background: rgba(79, 70, 229, 0.15);
      color: #a5b4fc;
      border-color: var(--accent);
    }
    .step-pill.done {
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.3);
    }
    .progress-bar-bg {
      height: 4px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 2px;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.25s ease;
    }
    .status-detail {
      font-size: 0.72rem;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .download-btn {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      font-weight: 600;
      font-size: 0.8rem;
      padding: 9px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .download-btn:hover {
      background: rgba(16, 185, 129, 0.2);
    }

    /* Stage Area */
    .stage-area {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      background: #080a0d;
      position: relative;
    }
    .stage-header {
      padding: 6px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
    }
    .view-mode-tabs {
      display: flex;
      background: #0d1016;
      padding: 2px;
      border-radius: 5px;
      border: 1px solid var(--border);
      gap: 1px;
    }
    .view-tab {
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.74rem;
      font-weight: 500;
      border: none;
      background: transparent;
      color: var(--text-dim);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s;
    }
    .view-tab.active {
      background: var(--bg-card);
      color: var(--text-bright);
      font-weight: 600;
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
      background: #07090c;
    }
    .viewport-card:last-child {
      border-right: none;
    }
    .viewport-tag {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 4;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(17, 20, 26, 0.85);
      padding: 4px 9px;
      border-radius: 5px;
      border: 1px solid var(--border);
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--text);
    }
    .viewport-meta {
      font-size: 0.66rem;
      padding: 1px 5px;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-dim);
      font-family: 'JetBrains Mono', monospace;
    }

    /* Video player */
    #videoViewport {
      align-items: center;
      justify-content: center;
      background: #05070a;
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
      background: #080a0e;
    }
    #viewport3d {
      width: 100%;
      height: 100%;
      display: block;
    }
    .viewport-hints {
      position: absolute;
      bottom: 10px;
      right: 10px;
      z-index: 4;
      font-size: 0.66rem;
      color: rgba(255, 255, 255, 0.35);
      background: rgba(10, 12, 16, 0.8);
      padding: 3px 8px;
      border-radius: 4px;
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
      padding: 20px;
      pointer-events: none;
      z-index: 2;
    }
    .viewport-empty svg {
      stroke: rgba(255, 255, 255, 0.12);
      margin-bottom: 10px;
    }
    .viewport-empty h4 {
      font-size: 0.9rem;
      color: var(--text);
      margin-bottom: 3px;
      font-weight: 500;
    }
    .viewport-empty p {
      font-size: 0.74rem;
      max-width: 220px;
      line-height: 1.4;
    }

    /* Viewport HUD buttons */
    .viewport-hud {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 5;
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(17, 20, 26, 0.85);
      padding: 3px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .hud-btn {
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 500;
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-dim);
      display: inline-flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .hud-btn svg {
      width: 12px;
      height: 12px;
    }
    .hud-btn:hover {
      color: var(--text-bright);
      background: rgba(255, 255, 255, 0.04);
    }
    .hud-btn.active {
      color: #ffffff;
      background: var(--bg-card);
      border-color: rgba(255, 255, 255, 0.12);
    }

    /* Bottom Playback Bar */
    .sync-controller {
      background: var(--bg-panel);
      border-top: 1px solid var(--border);
      padding: 8px 16px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 10;
    }
    .timeline-container {
      display: flex;
      align-items: center;
      gap: 10px;
      position: relative;
    }
    .time-slider {
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      border-radius: 2px;
      background: #1e2430;
      outline: none;
      cursor: pointer;
    }
    .time-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #ffffff;
      cursor: pointer;
      border: 2px solid var(--accent);
      transition: transform 0.1s;
    }
    .time-slider::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }
    .controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ctrl-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .ctrl-btn {
      padding: 5px 9px;
      border-radius: 5px;
      font-size: 0.74rem;
      font-weight: 500;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .ctrl-btn:hover {
      background: var(--bg-card-hover);
      color: var(--text-bright);
      border-color: var(--border-hover);
    }
    .ctrl-btn.play-btn {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      padding: 5px 14px;
      font-weight: 600;
    }
    .ctrl-btn.play-btn:hover {
      background: var(--accent-hover);
    }
    .ctrl-btn.active {
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.2);
      background: var(--bg-card-hover);
    }
    .time-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 6px;
    }
    .frame-pill {
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-dim);
    }
    .speed-group {
      display: flex;
      gap: 1px;
      background: #0d1016;
      padding: 2px;
      border-radius: 5px;
      border: 1px solid var(--border);
    }
    .speed-group .ctrl-btn {
      padding: 2px 6px;
      font-size: 0.7rem;
      background: transparent;
      border: none;
      color: var(--text-dim);
    }
    .speed-group .ctrl-btn.active {
      background: var(--bg-card);
      color: var(--text-bright);
      font-weight: 600;
    }

    /* Section Header Link */
    .section-link-btn {
      font-size: 0.68rem;
      color: #818cf8;
      background: transparent;
      border: none;
      cursor: pointer;
      font-weight: 500;
      padding: 0;
      text-decoration: underline;
      text-underline-offset: 2px;
      transition: color 0.15s;
    }
    .section-link-btn:hover {
      color: #a5b4fc;
    }

    /* Specifications & Rig Guide Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    .modal-overlay.open {
      display: flex;
    }
    .modal-dialog {
      width: 720px;
      max-width: 95vw;
      max-height: 85vh;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 24px 48px rgba(0, 0, 0, 0.7);
      overflow: hidden;
      animation: modalFadeIn 0.15s ease-out;
    }
    @keyframes modalFadeIn {
      from { opacity: 0; transform: scale(0.98); }
      to { opacity: 1; transform: scale(1); }
    }
    .modal-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #0d1016;
    }
    .modal-title-group h3 {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 2px;
    }
    .modal-title-group p {
      font-size: 0.72rem;
      color: var(--text-dim);
    }
    .modal-close-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 1.25rem;
      cursor: pointer;
      padding: 2px 8px;
      border-radius: 4px;
      line-height: 1;
      transition: all 0.15s;
    }
    .modal-close-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-bright);
    }
    .modal-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--bg-card);
      padding: 0 16px;
      gap: 16px;
    }
    .modal-tab-btn {
      padding: 10px 0;
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      font-size: 0.76rem;
      font-weight: 500;
      color: var(--text-dim);
      cursor: pointer;
      transition: all 0.15s;
    }
    .modal-tab-btn:hover {
      color: var(--text-bright);
    }
    .modal-tab-btn.active {
      color: #818cf8;
      border-bottom-color: var(--accent);
      font-weight: 600;
    }
    .modal-body {
      padding: 16px 18px;
      overflow-y: auto;
      font-size: 0.78rem;
      line-height: 1.55;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .modal-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .modal-section h4 {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .spec-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.74rem;
      margin-top: 6px;
    }
    .spec-table th, .spec-table td {
      padding: 6px 10px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .spec-table th {
      color: var(--text-dim);
      font-weight: 600;
      background: rgba(255, 255, 255, 0.02);
    }
    .spec-table td {
      color: var(--text);
    }
    .spec-pill {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.68rem;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.06);
      color: #a5b4fc;
    }
    .spec-pill.success {
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
    }
    .spec-code {
      font-family: 'JetBrains Mono', monospace;
      background: #090c10;
      padding: 8px 12px;
      border-radius: 5px;
      border: 1px solid var(--border);
      color: #38bdf8;
      font-size: 0.74rem;
      display: block;
      margin-top: 6px;
    }
    .modal-footer {
      padding: 10px 18px;
      border-top: 1px solid var(--border);
      background: #0d1016;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--text-dim);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        vid2anim
      </div>
      <span class="version-tag">v0.1.0</span>
    </div>
    <div class="header-actions">
      <div id="deviceBadge" class="device-pill">CPU</div>
      <button type="button" class="header-link" onclick="openRigGuideModal('rigs')" title="View supported 3D models, armatures, and bone naming">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
        <span>Model &amp; Rig Specs</span>
      </button>
      <a class="header-link" href="/api/download/blender-addon" download="vid2anim_blender.py" title="Download Blender 3D Add-on (.py)">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        <span>Blender Add-on</span>
      </a>
      <a class="header-link" href="https://github.com/mehulcode12/vid2anim" target="_blank" rel="noopener noreferrer" title="View Source on GitHub">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        <span>GitHub</span>
      </a>
      <div id="syncBadge" style="display:none"><span id="syncText"></span></div>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <!-- Section: Media Upload -->
      <div class="section-card">
        <div class="section-title">Source Media</div>
        <div class="upload-zone" id="mediaZone" onclick="document.getElementById('mediaInput').click()">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          <p>Drop video or photo</p>
          <span>MP4, MOV, WEBM, JPG, PNG</span>
        </div>
        <div id="mediaPill" class="file-pill">
          <div class="file-pill-info">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            <span id="mediaName" class="file-pill-name"></span>
          </div>
          <button class="file-pill-btn" onclick="clearMedia(event)">Remove</button>
        </div>
        <input type="file" id="mediaInput" accept="video/mp4,video/quicktime,video/webm,image/jpeg,image/png" style="display:none">
      </div>

      <!-- Section: 3D Model Upload -->
      <div class="section-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div class="section-title">Target Character</div>
          <button type="button" class="section-link-btn" onclick="openRigGuideModal('rigs')" title="What models &amp; rigs are supported?">Supported Rigs ↗</button>
        </div>
        <div class="upload-zone" id="modelZone" onclick="document.getElementById('modelInput').click()">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
          <p>Drop 3D character</p>
          <span>GLB, GLTF, FBX, OBJ</span>
        </div>
        <div id="modelPill" class="file-pill">
          <div class="file-pill-info">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            <span id="modelName" class="file-pill-name"></span>
          </div>
          <button class="file-pill-btn" onclick="clearModel(event)">Remove</button>
        </div>
        <input type="file" id="modelInput" accept=".glb,.gltf,.fbx,.obj" style="display:none">
      </div>

      <!-- Section: Tracking Engine -->
      <div class="section-card">
        <div class="section-title">Tracking Engine</div>
        <div class="segmented-control">
          <button type="button" class="segment-btn active" id="engineCardMediapipe" onclick="selectEngine('mediapipe')">
            MediaPipe Pose
          </button>
          <button type="button" class="segment-btn" id="engineCardDwpose" onclick="selectEngine('dwpose')">
            DWPose WholeBody
          </button>
        </div>
        <input type="hidden" id="selectedEngine" value="mediapipe">

        <div style="display:flex; flex-direction:column; gap:5px; margin-top:8px;">
          <div style="font-size:0.7rem; color:var(--text-dim); font-weight:600;">Execution Device</div>
          <select id="deviceSelect">
            <option value="cpu" selected>CPU</option>
            <option value="cuda">CUDA (GPU)</option>
          </select>
        </div>
      </div>

      <!-- Action Button -->
      <button id="runBtn" class="primary" onclick="startAnimationPipeline()" disabled>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Animate Character
      </button>

      <!-- Pipeline Progress -->
      <div id="statusCard" class="status-card" style="display:none">
        <div class="status-header">
          <span id="statusTitle">Processing</span>
          <span id="statusPercent" style="color:var(--text-bright)">0%</span>
        </div>
        <div class="pipeline-stepper">
          <div id="stepPill1" class="step-pill">Ingest</div>
          <div id="stepPill2" class="step-pill">Extract</div>
          <div id="stepPill3" class="step-pill">Retarget</div>
          <div id="stepPill4" class="step-pill">Pack GLB</div>
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
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download Animated GLB
      </button>
    </div>

    <!-- Main Stage Viewport Area -->
    <div class="stage-area">
      <div class="stage-header">
        <div class="view-mode-tabs">
          <button class="view-tab active" id="tabSplit" onclick="setViewMode('split')">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"/><rect x="13" y="3" width="8" height="18" rx="2"/></svg>
            Side-by-Side
          </button>
          <button class="view-tab" id="tab3d" onclick="setViewMode('3d')">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
            3D Character
          </button>
          <button class="view-tab" id="tabVideo" onclick="setViewMode('video')">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Reference Video
          </button>
        </div>

        <div class="stage-tools">
          <select id="animSelect" onchange="switchAnimation(this.value)" style="display:none; font-size:0.72rem; padding:3px 7px;"></select>
          <button class="ctrl-btn" onclick="resetCamera()" title="Reset 3D Camera [R]">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
            Reset View
          </button>
        </div>
      </div>

      <!-- Viewports Grid -->
      <div class="viewports-container mode-split" id="viewportsContainer">
        <!-- Video Viewport -->
        <div class="viewport-card" id="videoViewport">
          <div class="viewport-tag">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Reference Video
            <span id="videoMetaBadge" class="viewport-meta">No media</span>
          </div>
          <div id="videoEmptyState" class="viewport-empty">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
            <h4>Reference Footage</h4>
            <p>Upload or drag a video or photo onto the left panel to preview playback</p>
          </div>
          <video id="sourceVideo" playsinline muted style="display:none"></video>
          <img id="sourceImage" alt="Reference Image">
        </div>

        <!-- 3D Viewport -->
        <div class="viewport-card" id="modelViewport">
          <div class="viewport-tag">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            3D Character
            <span id="modelMetaBadge" class="viewport-meta">Waiting for model</span>
          </div>

          <!-- Floating Viewport HUD -->
          <div class="viewport-hud">
            <button class="hud-btn" id="btnSkeleton" onclick="toggleSkeleton()" title="Toggle Skeleton Overlay [B]">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="3"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="12" y1="12" x2="6" y2="10"/><line x1="12" y1="12" x2="18" y2="10"/><line x1="12" y1="16" x2="8" y2="21"/><line x1="12" y1="16" x2="16" y2="21"/></svg>
              Skeleton
            </button>
            <button class="hud-btn active" id="btnGrid" onclick="toggleGrid()" title="Toggle Floor Grid [G]">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
              Grid
            </button>
            <button class="hud-btn" id="btnWireframe" onclick="toggleWireframe()" title="Toggle Wireframe">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
              Wireframe
            </button>
          </div>

          <div id="modelEmptyState" class="viewport-empty">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            <h4>3D Character Viewport</h4>
            <p>Upload or drag a rigged 3D character (GLB, GLTF, FBX) onto the left panel</p>
          </div>
          <div id="viewport3d"></div>
          <div class="viewport-hints">Rotate: Left Drag | Pan: Right Drag | Zoom: Scroll | Play: Space</div>
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
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              <span>Play</span>
            </button>
            <button class="ctrl-btn" onclick="stepFrame(-1)" title="Previous Frame [Left Arrow]">
              -1F
            </button>
            <button class="ctrl-btn" onclick="stepFrame(1)" title="Next Frame [Right Arrow]">
              +1F
            </button>
            <div class="time-display">
              <span id="timecodeText">00:00.00 / 00:00.00</span>
              <span id="framePill" class="frame-pill">Frame 0 / 0</span>
            </div>
          </div>

          <!-- Speed & Loop -->
          <div class="ctrl-group">
            <div class="speed-group">
              <button class="ctrl-btn" onclick="setSpeed(0.5)">0.5x</button>
              <button class="ctrl-btn active" id="spd1x" onclick="setSpeed(1.0)">1.0x</button>
              <button class="ctrl-btn" onclick="setSpeed(1.5)">1.5x</button>
              <button class="ctrl-btn" onclick="setSpeed(2.0)">2.0x</button>
            </div>
            <button id="loopBtn" class="ctrl-btn active" onclick="toggleLoop()" title="Toggle Looping [L]">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
              Loop
            </button>
            <button id="muteBtn" class="ctrl-btn active" onclick="toggleMute()" title="Toggle Audio [M]">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
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
        } else if (e.code === 'Escape') {
          closeRigGuideModal();
        }
      });
    }

    // Model & Rigging Guide Modal Handlers
    function openRigGuideModal(tabId) {
      document.getElementById('rigGuideModal').classList.add('open');
      if (tabId) switchGuideTab(tabId);
    }

    function closeRigGuideModal() {
      document.getElementById('rigGuideModal').classList.remove('open');
    }

    function switchGuideTab(tabId) {
      document.querySelectorAll('.modal-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.guide-tab-pane').forEach(p => p.style.display = 'none');
      const btn = document.getElementById('tabBtn_' + tabId);
      const pane = document.getElementById('tabPane_' + tabId);
      if (btn) btn.classList.add('active');
      if (pane) pane.style.display = 'block';
    }

    // Diagnostics & File Handlers
    fetch('/api/diagnostics')
      .then(r => r.json())
      .then(d => {
        const badge = document.getElementById('deviceBadge');
        badge.innerText = d.cuda_available ? 'CUDA' : 'CPU';
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

    // HTML5 Drag and Drop Handlers
    function wireDragDrop(zoneId, handleFn) {
      const el = document.getElementById(zoneId);
      if (!el) return;
      ['dragenter', 'dragover'].forEach(n => {
        el.addEventListener(n, (e) => {
          e.preventDefault();
          e.stopPropagation();
          el.classList.add('dragover');
        });
      });
      ['dragleave', 'drop'].forEach(n => {
        el.addEventListener(n, (e) => {
          e.preventDefault();
          e.stopPropagation();
          el.classList.remove('dragover');
        });
      });
      el.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
          handleFn(dt.files[0]);
        }
      });
    }
    wireDragDrop('mediaZone', handleMediaFile);
    wireDragDrop('modelZone', handleModelFile);

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
          let errMsg = 'Animation pipeline failed';
          try {
            const err = await response.json();
            errMsg = err.error || errMsg;
          } catch (_) {
            errMsg = `Server error (HTTP ${response.status})`;
          }
          throw new Error(errMsg);
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
        statusText.innerText = `Animation ready${boneNotice}`;
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
  <!-- Model & Rigging Specifications Modal -->
  <div class="modal-overlay" id="rigGuideModal" onclick="if(event.target===this) closeRigGuideModal()">
    <div class="modal-dialog">
      <div class="modal-header">
        <div class="modal-title-group">
          <h3>3D Model &amp; Rigging Specifications</h3>
          <p>Supported formats, skeletal armature standards, and bone naming conventions</p>
        </div>
        <button class="modal-close-btn" onclick="closeRigGuideModal()" title="Close (Esc)">&times;</button>
      </div>

      <div class="modal-tabs">
        <button type="button" class="modal-tab-btn active" id="tabBtn_rigs" onclick="switchGuideTab('rigs')">Supported Formats &amp; Rigs</button>
        <button type="button" class="modal-tab-btn" id="tabBtn_mapping" onclick="switchGuideTab('mapping')">Bone Mapping &amp; Fingers</button>
        <button type="button" class="modal-tab-btn" id="tabBtn_checklist" onclick="switchGuideTab('checklist')">Requirements &amp; CLI</button>
      </div>

      <div class="modal-body">
        <!-- Pane 1: Supported Formats & Rigs -->
        <div class="guide-tab-pane" id="tabPane_rigs">
          <div class="modal-section">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
              Supported 3D File Formats
            </h4>
            <table class="spec-table">
              <thead>
                <tr>
                  <th>Format</th>
                  <th>Status</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><span class="spec-pill success">.GLB</span></td>
                  <td>Recommended</td>
                  <td>glTF 2.0 Binary container. Embeds mesh, skeleton hierarchy, and vertex skin weights in one file.</td>
                </tr>
                <tr>
                  <td><span class="spec-pill">.GLTF</span></td>
                  <td>Supported</td>
                  <td>glTF 2.0 JSON format (with separate or embedded .bin buffers).</td>
                </tr>
                <tr>
                  <td><span class="spec-pill">.FBX</span></td>
                  <td>Supported</td>
                  <td>Humanoid FBX assets with standard skeletal armatures.</td>
                </tr>
                <tr>
                  <td><span class="spec-pill" style="color:var(--text-dim)">.OBJ</span></td>
                  <td>Static Only</td>
                  <td>Static geometry only. OBJ does not support skeletons or skin weights (requires rigging first).</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="modal-section" style="margin-top:12px;">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="3"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="12" y1="12" x2="6" y2="10"/><line x1="12" y1="12" x2="18" y2="10"/><line x1="12" y1="16" x2="8" y2="21"/><line x1="12" y1="16" x2="16" y2="21"/></svg>
              Recognized Armature &amp; Rig Ecosystems
            </h4>
            <p style="color:var(--text-dim); margin-bottom:8px;">vid2anim includes an intelligent NLP semantic matcher and kinematic graph solver that automatically identifies these industry rig conventions:</p>
            <table class="spec-table">
              <thead>
                <tr>
                  <th>Rig Standard</th>
                  <th>Typical Bone Naming</th>
                  <th>Detection</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Adobe Mixamo</strong></td>
                  <td><code>mixamorig:Hips</code>, <code>mixamorig:LeftArm</code>, <code>mixamorig:RightUpLeg</code></td>
                  <td><span class="spec-pill success">Native Auto-Detect</span></td>
                </tr>
                <tr>
                  <td><strong>VRM / Ready Player Me</strong></td>
                  <td><code>J_Bip_C_Hips</code>, <code>J_Bip_L_UpperArm</code>, <code>J_Bip_L_Hand</code></td>
                  <td><span class="spec-pill success">Native Auto-Detect</span></td>
                </tr>
                <tr>
                  <td><strong>Blender Rigify</strong></td>
                  <td><code>def-spine</code>, <code>def-upper_arm.l</code>, <code>def-thigh.r</code></td>
                  <td><span class="spec-pill success">Native Auto-Detect</span></td>
                </tr>
                <tr>
                  <td><strong>Unreal Engine (UE4/UE5)</strong></td>
                  <td><code>pelvis</code>, <code>upperarm_l</code>, <code>lowerarm_l</code>, <code>thigh_r</code></td>
                  <td><span class="spec-pill success">Native Auto-Detect</span></td>
                </tr>
                <tr>
                  <td><strong>Character Creator / Daz</strong></td>
                  <td><code>CC_Base_Pelvis</code>, <code>CC_Base_L_Upperarm</code>, <code>b_hips</code></td>
                  <td><span class="spec-pill success">Native Auto-Detect</span></td>
                </tr>
                <tr>
                  <td><strong>Generic Humanoid</strong></td>
                  <td>Snake_case, CamelCase, dot notation, or Japanese Kanji (<code>腰</code>, <code>腕</code>)</td>
                  <td><span class="spec-pill success">Semantic Matcher</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Pane 2: Bone Mapping & Fingers -->
        <div class="guide-tab-pane" id="tabPane_mapping" style="display:none;">
          <div class="modal-section">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              Canonical Humanoid Hierarchy (22 Primary Bones)
            </h4>
            <p style="color:var(--text-dim); margin-bottom:8px;">Extracted 3D poses are transferred via reference forward kinematics onto this core hierarchy:</p>
            <table class="spec-table">
              <thead>
                <tr>
                  <th>Anatomical Region</th>
                  <th>Canonical Joints</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Pelvis / Root</strong></td>
                  <td><code>hips</code> (acts as motion root with 6-DOF translation &amp; rotation)</td>
                </tr>
                <tr>
                  <td><strong>Spine &amp; Torso</strong></td>
                  <td><code>spine</code> &rarr; <code>spine1</code> &rarr; <code>chest</code></td>
                </tr>
                <tr>
                  <td><strong>Head &amp; Neck</strong></td>
                  <td><code>neck</code> &rarr; <code>head</code></td>
                </tr>
                <tr>
                  <td><strong>Left Upper Limb</strong></td>
                  <td><code>left_shoulder</code> &rarr; <code>left_arm</code> &rarr; <code>left_forearm</code> &rarr; <code>left_hand</code></td>
                </tr>
                <tr>
                  <td><strong>Right Upper Limb</strong></td>
                  <td><code>right_shoulder</code> &rarr; <code>right_arm</code> &rarr; <code>right_forearm</code> &rarr; <code>right_hand</code></td>
                </tr>
                <tr>
                  <td><strong>Left Lower Limb</strong></td>
                  <td><code>left_upleg</code> &rarr; <code>left_leg</code> &rarr; <code>left_foot</code> &rarr; <code>left_toe</code></td>
                </tr>
                <tr>
                  <td><strong>Right Lower Limb</strong></td>
                  <td><code>right_upleg</code> &rarr; <code>right_leg</code> &rarr; <code>right_foot</code> &rarr; <code>right_toe</code></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="modal-section" style="margin-top:12px;">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
              Fingers &amp; Procedural Helpers
            </h4>
            <p style="color:var(--text-dim); margin-bottom:6px;">If your character rig has finger bones or twist helpers, vid2anim automatically animates them:</p>
            <ul style="color:var(--text); padding-left:18px; line-height:1.6; font-size:0.74rem;">
              <li><strong>Independent 5-Digit Finger Cascade</strong>: Proximal, intermediate, and distal phalanges for Thumb, Index, Middle, Ring, and Pinky are articulated with natural anatomical flexion based on tracked hand gestures.</li>
              <li><strong>Twist &amp; Share Bones</strong>: Arm, forearm, and thigh twist helper bones are driven with proportional fractional rotations to prevent mesh pinching at joints.</li>
              <li><strong>IK &amp; Accessory Bones</strong>: Detached IK foot nodes and hip attachments follow character ground translations cleanly.</li>
            </ul>
          </div>
        </div>

        <!-- Pane 3: Requirements & CLI -->
        <div class="guide-tab-pane" id="tabPane_checklist" style="display:none;">
          <div class="modal-section">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
              Character Preparation Checklist
            </h4>
            <div style="display:flex; flex-direction:column; gap:8px; font-size:0.74rem; color:var(--text);">
              <div>&bull; <strong>Mesh Must Be Skinned</strong>: In Blender or your 3D software, the mesh object must have an Armature modifier with vertex skinning weights (glTF <code>skins</code> definition).</div>
              <div>&bull; <strong>Bind Pose (T-Pose / A-Pose)</strong>: The default pose of the character should ideally be a standard T-Pose or A-Pose. vid2anim automatically measures rest arm angles (0&deg; to 65&deg;) and compensates.</div>
              <div>&bull; <strong>Orientation</strong>: Model should be upright (Y-Up axis) facing forwards (+Z or -Z).</div>
              <div>&bull; <strong>Scale</strong>: Real-world metric dimensions (approx. 1.6m to 1.9m tall for adult humanoids).</div>
            </div>
          </div>

          <div class="modal-section" style="margin-top:12px;">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
              CLI Model Rig Inspector
            </h4>
            <p style="color:var(--text-dim);">You can test and inspect any 3D asset from the command line before retargeting:</p>
            <code class="spec-code">vid2anim inspect path/to/character.glb</code>
            <p style="color:var(--text-dim); margin-top:6px;">This command analyzes the armature, prints detected rig type, mapped joint count, rest arm angle, and warns about any missing core bones.</p>
          </div>

          <div class="modal-section" style="margin-top:12px;">
            <h4>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
              Blender 3D Native Bridge Add-on
            </h4>
            <p style="color:var(--text-dim);">Animate characters directly in Blender 3D Viewport without manual file conversions:</p>
            <div style="display:flex; align-items:center; gap:10px; margin-top:8px;">
              <a href="/api/download/blender-addon" download="vid2anim_blender.py" class="ctrl-btn active" style="text-decoration:none; display:inline-flex; align-items:center; gap:6px;">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Download vid2anim_blender.py
              </a>
              <span style="color:var(--text-dim); font-size:0.7rem;">Works in Blender 3.0+ &amp; 4.0+</span>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <span>Press <kbd style="background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:3px; font-family:inherit;">Esc</kbd> or click outside to dismiss</span>
        <button type="button" class="ctrl-btn active" onclick="closeRigGuideModal()" style="padding:4px 14px;">Close</button>
      </div>
    </div>
  </div>
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

        elif parsed_path.path == "/api/download/blender-addon":
            addon_path = Path(__file__).resolve().parent.parent / "plugins" / "blender" / "vid2anim_blender.py"
            if not addon_path.exists():
                addon_path = Path("plugins/blender/vid2anim_blender.py").resolve()

            if addon_path.exists():
                data = addon_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/x-python")
                self.send_header("Content-Disposition", 'attachment; filename="vid2anim_blender.py"')
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

            if not media_info.get("data") or not model_info.get("data"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty file received for media or model"}).encode("utf-8"))
                return

            device_str = fields.get("device", "auto")
            device_str = device_str.strip().lower() if isinstance(device_str, str) else "auto"
            engine_str = fields.get("engine", "mediapipe")
            engine_str = engine_str.strip().lower() if isinstance(engine_str, str) else "mediapipe"

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
