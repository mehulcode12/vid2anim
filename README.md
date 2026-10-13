# vid2anim 🎬 ➡️ 🕺

> **Animate ANY 3D character model from a photo or a video (≤ 30s) using AI motion capture and universal retargeting.**  
> **100% CPU-compatible by default — with optional NVIDIA CUDA GPU acceleration.**

---

## 🌟 Highlights

- ⚡ **Runs on CPU by Default:** Don't have an expensive NVIDIA RTX GPU? No problem! `vid2anim` runs an optimized CPU-friendly computer vision pipeline so anyone can use it on a standard laptop.
- 🚀 **Optional GPU Acceleration:** If you have an NVIDIA GPU with CUDA, simply pass `--device cuda` for turbo-charged inference.
- 🧍 **Any 3D Model:**
  - **Pre-rigged Humanoid Models:** Auto-detects bone hierarchies for **Mixamo**, **VRM (VTubers)**, **Ready Player Me**, **Blender Rigify**, and **Unreal Engine Mannequins**.
  - **Static Unrigged Meshes:** Built-in **Experimental Auto-Rigging Engine** that fits a humanoid skeleton and computes smooth skinning weights for raw meshes.
- 📹 **Flexible Input:** Upload a single photo (`.jpg`, `.png`) to transfer a static 3D pose, or a short action video (`.mp4`, up to 30s) for full motion sequences.
- 📦 **Standard Output (.glb):** Exports production-ready, textured, skinned glTF 2.0 (`.glb`) files that open instantly in **Blender, Unity, Unreal Engine, Three.js, and Windows 3D Viewer**.
- 🔄 **Motion Caching:** Decouples video extraction from 3D retargeting. Extract once, then retarget onto 10 different 3D models in seconds.

---

## 🏗️ Architecture

```
                      ┌────────────────────────────────────────┐
                      │   User Input: Photo or Video (≤ 30s)   │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │        Motion Extractor Engine         │
                      │  - CPU Mode (Fast real-time landmarks) │
                      │  - GPU Mode (CUDA-accelerated)         │
                      │  → Saves cached `motion_cache.npz`     │
                      └──────────────────┬─────────────────────┘
                                         │
 ┌─────────────────────────┐             │
 │   User's 3D Character   │             │
 │ (.glb, .gltf, .fbx, obj)│             │
 └───────────┬─────────────┘             │
             ▼                           │
 ┌─────────────────────────┐             │
 │   Rig / Bone Detector   │             │
 │ - Pre-rigged (Mixamo,   │             │
 │   VRM, Rigify, Custom)? │             │
 │ - Or Static Mesh?       │             │
 │   (Auto-Rig Fallback)   │             │
 └───────────┬─────────────┘             │
             ▼                           ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 Universal Retargeter Engine                 │
 │  - Swing-Twist vector aiming & T-pose calibration           │
 │  - Quaternion sign continuity (prevents 360° flip pops)     │
 │  - Root motion translation scaling                          │
 └───────────────────────────────┬─────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     Exporter & Output                       │
 │  - Skinned Animated 3D Model: `.glb`                        │
 │  - Universal MoCap File: `.bvh`                             │
 └─────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/vid2anim.git
cd vid2anim
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

**For standard CPU mode (Default):**
```bash
pip install -e .
```

**For NVIDIA GPU acceleration (Optional):**
```bash
pip install -e .[gpu]
```

### 3. Verify System & Hardware

Run the built-in diagnostic doctor:
```bash
vid2anim doctor
```

It will print your hardware details, CPU core count, whether a CUDA GPU is present, and confirm FFmpeg readiness.

---

## 🚀 Quickstart & Commands

### 1. Check a 3D Model's Rig

Before animating, inspect any 3D asset to verify its bone structure:
```bash
vid2anim inspect assets/character.glb
```

### 2. End-to-End Run (Photo or Video ➡️ Animated 3D Model)

Animate your 3D character directly from a video:
```bash
# On CPU (default):
vid2anim run --input dance.mp4 --model character.glb --output animated.glb

# On GPU (if available):
vid2anim run --input dance.mp4 --model character.glb --device cuda
```

Animate a static pose from a photo:
```bash
vid2anim run --input photo.jpg --model character.glb --output posed_character.glb
```

### 3. Fast Workflow: Extract Once, Retarget to Multiple Characters

Extract the 3D motion once:
```bash
vid2anim extract --input martial_arts.mp4 --output outputs/motion.npz
```

Now retarget that exact motion onto 5 different characters instantly without waiting:
```bash
vid2anim retarget --motion outputs/motion.npz --model robot.glb --output robot_action.glb
vid2anim retarget --motion outputs/motion.npz --model wizard.glb --output wizard_action.glb
```

---

## 🌐 Interactive Web UI & 3D Viewport

Launch the built-in local web application directly from the CLI:
```bash
vid2anim serve --port 8000
```
Open **`http://localhost:8000`** in any web browser to:
- 🖱️ **Drag and drop** your video/photo and 3D character (`.glb`, `.fbx`, `.obj`).
- ⚡ **Toggle CPU vs GPU** inference.
- 🕺 **Preview animation live** in the interactive Three.js 3D viewport (orbit, rotate, play/pause).
- 💾 **Download** the resulting animated `.glb` with one click!

---

## ⚡ Automation & Streaming Hooks

The CLI features `--progress-jsonl <file.jsonl>` which emits structured event milestones:
```bash
vid2anim run --input clip.mp4 --model bot.glb --progress-jsonl status.jsonl
```
This enables effortless integration with:
- **FastAPI / REST backend services**
- **React / Next.js / Three.js web viewers**
- **Blender 4.2+ extensions**

---

## ⚖️ License

MIT License. Free for open-source and commercial use.
