# vid2anim

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blender 3.0+](https://img.shields.io/badge/Blender-3.0%20--%204.2+-E87D0D.svg?style=flat&logo=blender&logoColor=white)](https://www.blender.org/)
[![Hardware: CPU & CUDA](https://img.shields.io/badge/hardware-CPU%20%7C%20CUDA-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://github.com/mehulcode12/vid2anim)
[![Tests: Passing](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](https://github.com/mehulcode12/vid2anim)

> **Animate ANY 3D character model from video or photos using AI motion capture and universal retargeting.**  
> **100% CPU-compatible by default — with optional NVIDIA CUDA GPU acceleration.**

---

## Highlights

- **Runs on CPU by Default:** No expensive GPU required. Runs an optimized, lightweight computer vision pipeline out of the box on standard laptops and workstations.
- **Optional CUDA GPU Acceleration:** Pass `--device cuda` for high-throughput batch and real-time processing on NVIDIA hardware.
- **Dual AI Engine Architecture:**
  - **MediaPipe Pose:** Ultra-fast CPU landmark tracking (19 canonical body joints). Instant setup with zero model downloads.
  - **DWPose WholeBody:** High-fidelity 133-landmark tracking covering body pose, facial expression orientation, and 5-digit finger articulation.
- **Universal Humanoid Retargeting:**
  - **Pre-rigged Humanoid Models:** Auto-detects bone hierarchies for **Mixamo**, **VRM (VTubers)**, **Ready Player Me**, **Blender Rigify**, **Unreal Engine Mannequins (UE4/UE5)**, and **Daz3D / Character Creator**.
  - **Independent 5-Digit Finger Articulation:** Dynamic anatomical flexion cascade for Thumb, Index, Middle, Ring, and Pinky digits with procedural twist helpers.
  - **Static Unrigged Meshes:** Built-in **Auto-Rigging Engine** that fits a humanoid skeleton and computes smooth heat-diffusion skinning weights for raw meshes (`.glb`, `.obj`).
  - **SO(3) Lie Group Kinematics:** Closed-form basis transfer ($C D C^T$) and quaternion shortest-arc SLERP ensuring zero 360° flip pops or Gimbal lock.
- **Blender 3D Bridge Add-on:** Dedicated native add-on (`plugins/blender/vid2anim_blender.py`) for Blender 3.0 to 4.2+ LTS. 1-click retargeting right in the 3D Viewport with zero third-party pip dependencies in Blender.
- **Production-Ready Standards:** Exports strictly compliant glTF 2.0 (`.glb`) files with 4-byte chunk alignment, ready for **Blender, Unreal Engine, Unity, Three.js, Godot, and Windows 3D Viewer**, plus standard universal `.bvh` mocap files.
- **Interactive Web UI & Real-Time Viewport:** Built-in dark workstation interface with Three.js orbit viewport, synchronized video/3D scrubbing, frame stepping, and model rig inspection.

---

## Architecture

```
                      ┌────────────────────────────────────────┐
                      │    User Input: Photo or Action Video   │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │        Motion Extractor Engine         │
                      │  - MediaPipe Pose (CPU Fast)           │
                      │  - DWPose WholeBody (133 landmarks)    │
                      │  - CUDA GPU Acceleration (Optional)    │
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
 │ - Mixamo, VRM, Rigify,  │             │
 │   UE Mannequin, Daz3D   │             │
 │ - Static Mesh Fallback  │             │
 │   (Auto-Rig Skinning)   │             │
 └───────────┬─────────────┘             │
             ▼                           ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 Universal Retargeter Engine                 │
 │  - Canonical Humanoid Forward Kinematics (FK)               │
 │  - Closed-form Lie algebra basis transfer (C D C^T)         │
 │  - Independent 5-digit finger flexion cascade               │
 │  - Procedural twist bone distribution                       │
 │  - Quaternion sign continuity (prevents 360° flip pops)     │
 │  - Ground contact alignment & root translation scaling      │
 └───────────────────────────────┬─────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     Exporter & Output                       │
 │  - Skinned Animated 3D Model: `.glb` (glTF 2.0 Compliant)   │
 │  - Universal MoCap File: `.bvh`                             │
 │  - Native Blender Bridge: Bakes directly to Action timeline │
 └─────────────────────────────────────────────────────────────┘
```

---

## Supported 3D Rig Profiles & Specifications

`vid2anim` automatically recognizes bone naming conventions and hierarchies across all standard industry formats:

| Rig Format | Identifier / Prefix | Fingers Supported | Default Rest Pose | Target Environments |
|:---|:---|:---:|:---:|:---|
| **Mixamo** | `mixamorig:Hips`, `mixamorig:LeftArm` | All 5 Digits | T-Pose | Web, Unity, Blender, Games |
| **VRM / Ready Player Me** | `J_Bip_C_Hips`, `hips`, `leftUpperArm` | All 5 Digits | T-Pose / A-Pose | VTubers, VRChat, Unity, Three.js |
| **Blender Rigify** | `DEF-spine`, `DEF-upper_arm.L` | All 5 Digits | T-Pose / A-Pose | Blender 3D production pipeline |
| **Unreal Engine Mannequin** | `pelvis`, `upperarm_l`, `hand_l` | All 5 Digits | A-Pose (30°/45°) | Unreal Engine 4 / 5, Cine, Games |
| **Daz3D / Character Creator** | `hip`, `lShldr`, `lForeArm` | All 5 Digits | T-Pose | Film, Cinematics, Game dev |
| **Static Mesh (Auto-Rig)** | No bones detected | Procedural | Auto-Fitted | Any unrigged `.glb`, `.obj` mesh |

To verify any 3D model's armature before retargeting, run:
```bash
vid2anim inspect character.glb
```

---

## Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/mehulcode12/vid2anim.git
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

**For NVIDIA CUDA GPU acceleration (Optional):**
```bash
pip install -e .[gpu]
```

**For complete developer & test suite dependencies:**
```bash
pip install -e .[full,dev]
```

### 3. Verify System & Hardware

Run the built-in diagnostic doctor:
```bash
vid2anim doctor
```

Output confirms CPU core count, CUDA availability, OpenCV / FFmpeg video backend, and recommended compute device.

---

## Quickstart & CLI Reference

### 1. Inspect a 3D Character Rig

```bash
vid2anim inspect character.glb
```

Prints model format, skeleton hierarchy, rig profile (Mixamo, VRM, Rigify, UE5, etc.), and canonical joint bindings.

### 2. End-to-End Animation (Media ➡️ Animated 3D Model)

Animate any 3D character directly from an action video:
```bash
# Standard CPU mode (Fast MediaPipe tracking)
vid2anim run --input dance.mp4 --model character.glb --output animated.glb

# WholeBody 133-point tracking with dense fingers & GPU
vid2anim run --input dance.mp4 --model character.glb --engine dwpose --device cuda

# Static pose transfer from a photograph
vid2anim run --input pose.jpg --model character.glb --output posed.glb
```

### 3. Decoupled Workflow: Extract Once, Retarget to Multiple Characters

Extract the 3D motion once to a reusable compressed `.npz`:
```bash
vid2anim extract --input martial_arts.mp4 --output motion.npz
```

Instantly retarget that exact motion onto multiple characters without re-running AI extraction:
```bash
vid2anim retarget --motion motion.npz --model robot.glb --output robot_action.glb
vid2anim retarget --motion motion.npz --model warrior.glb --output warrior_action.glb
vid2anim retarget --motion motion.npz --model vtuber.vrm --output vtuber_action.glb
```

### 4. CLI Arguments Reference

| Command | Flag | Description | Default |
|:---|:---|:---|:---|
| `run`, `extract` | `--input`, `-i` | Path to reference video (`.mp4`, `.mov`, `.webm`) or photo (`.jpg`, `.png`) | *Required* |
| `run`, `retarget` | `--model`, `-m` | Path to target 3D model (`.glb`, `.gltf`, `.fbx`, `.obj`) | *Required* |
| `run`, `retarget` | `--output`, `-o` | Output file path for animated `.glb` | `outputs/<name>_animated.glb` |
| `run` | `--bvh` | Optional path to export standard `.bvh` motion capture file | `None` |
| `run`, `extract` | `--device` | Compute device (`cpu`, `cuda`, `auto`) | `auto` |
| `run`, `extract` | `--engine` | AI tracking engine (`mediapipe`, `dwpose`) | `mediapipe` |
| `run`, `extract` | `--max-duration`| Max clip duration in seconds (up to 600s) | `30.0` |
| `run` | `--rest-pose` | Target rest pose calibration (`auto`, `t_pose`, `a_pose_30`, `a_pose_45`) | `auto` |
| `run` | `--no-smooth` | Disable temporal anti-jitter filter | `False` |
| `run` | `--progress-jsonl` | Filepath for streaming real-time JSONL progress events | `None` |

---

## Blender 3D Bridge Add-on

`vid2anim` includes a dedicated native Blender add-on for 1-click retargeting directly inside Blender 3.0, 3.6 LTS, 4.0, 4.1, and 4.2+ LTS.

### Installation

1. Start the local `vid2anim` server:
   ```bash
   vid2anim serve --port 8000
   ```
2. In Blender, navigate to **Edit &rarr; Preferences &rarr; Add-ons**.
3. Click **Install...** (top right) and select `plugins/blender/vid2anim_blender.py` (or click **Download Blender Add-on** in the Web UI).
4. Enable the checkbox for **Animation: vid2anim Blender Bridge**.

### Usage in Blender

1. In the 3D Viewport, press <kbd>N</kbd> to open the sidebar and click the **`vid2anim`** tab.
2. Select your character armature in the scene.
3. Under **Reference Media**, select your reference video or photo.
4. Choose your tracking engine (**MediaPipe Pose** or **DWPose WholeBody**) and compute device.
5. Click **"Animate Selected Armature"**.
6. The animation keyframes are baked directly into a standard Blender `Action` ready for editing in the Dope Sheet or Non-Linear Animation (NLA) editor.

---

## Interactive Web UI & 3D Viewport

Launch the workstation interface directly from the CLI:

```bash
vid2anim serve --host 127.0.0.1 --port 8000
```

Open **`http://localhost:8000`** in your browser to access:
- **Interactive 3D Viewport:** Three.js orbit viewport with smooth lighting, grid, bone count indicators, and wireframe toggles.
- **Side-by-Side Playback:** Synchronized video and 3D character scrubbing with sub-frame timecode display and frame-stepping buttons (<kbd>&larr;</kbd> / <kbd>&rarr;</kbd>).
- **Engine Switcher:** Seamlessly switch between **MediaPipe Pose** (CPU real-time) and **DWPose WholeBody** (dense 133-point body, face, and fingers).
- **3D Model & Rigging Specifications Modal:** Comprehensive in-app guide detailing supported formats, naming tables, finger articulation, and CLI tips.
- **One-Click Export:** Download the animated `.glb` or copy CLI commands instantly.

---

## Python API Usage

You can also integrate `vid2anim` directly into your Python scripts and pipelines:

```python
from pathlib import Path
from vid2anim import PipelineConfig, Vid2AnimPipeline

# Configure the retargeting pipeline
config = PipelineConfig(
    input_media=Path("dance.mp4"),
    target_model=Path("cyborg.glb"),
    output_glb=Path("output/cyborg_animated.glb"),
    device="auto",            # "cpu" or "cuda"
    engine="dwpose",          # "mediapipe" or "dwpose"
    smooth=True,              # Anti-jitter filter
    max_duration_sec=60.0,
)

# Run end-to-end animation
pipeline = Vid2AnimPipeline(config)
result = pipeline.execute()

if result.success:
    print(f"Animation saved to: {result.output_glb}")
    print(f"Processed {result.frame_count} frames in {result.elapsed_time_sec:.2f}s")
else:
    print(f"Failed: {result.error_message}")
```

---

## Testing

Run the comprehensive unit and integration test suite:

```bash
pytest -v
```

All 44 test cases cover:
- SO(3) Lie group kinematics & quaternion continuity
- glTF 2.0 4-byte chunk padding and binary GLB encoding
- Rig hierarchy classification (Mixamo, VRM, Rigify, UE4/5, Daz3D)
- Independent 5-digit finger flexion cascade
- REST API endpoints and Blender add-on communication
- Video decoding edge cases and motion cache I/O

---

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request:
1. Fork the repository: `https://github.com/mehulcode12/vid2anim`
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Free for open-source and commercial use.
