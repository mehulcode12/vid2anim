<div align="center">

# vid2anim

### AI Motion Capture & Universal 3D Character Retargeting Studio

**Turn any video or photo into production-ready 3D character animations.**  
Works with **Mixamo, VRM / VTubers, Blender Rigify, Unreal Engine Mannequins, Daz3D**, and even **raw unrigged meshes**.

[![Python 3.10 - 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blender 3.0 - 4.2+ LTS](https://img.shields.io/badge/Blender-3.0%20to%204.2+-E87D0D.svg?style=flat&logo=blender&logoColor=white)](https://www.blender.org/)
[![Hardware: 100% CPU Ready](https://img.shields.io/badge/Hardware-100%25%20CPU%20Ready%20(No%20GPU%20Required)-00C853.svg?style=flat)](https://github.com/mehulcode12/vid2anim)
[![Output: GLB & BVH](https://img.shields.io/badge/Output-glTF%202.0%20(.glb)%20%7C%20BVH-673AB7.svg?style=flat)](https://github.com/mehulcode12/vid2anim)
[![Tests: 49 Passing](https://img.shields.io/badge/Tests-49%20passed-brightgreen.svg)](https://github.com/mehulcode12/vid2anim)
[![CI Test Suite](https://github.com/mehulcode12/vid2anim/actions/workflows/ci.yml/badge.svg)](https://github.com/mehulcode12/vid2anim/actions/workflows/ci.yml)

[Key Features](#-what-makes-it-built-for-3d-artists) •
[Three Ways to Use](#-three-ways-to-use-vid2anim) •
[Supported 3D Rigs](#-supported-character-rigs--formats) •
[2-Minute Quickstart](#-2-minute-quickstart) •
[Blender Add-on Guide](#-blender-3d-bridge-add-on) •
[Animator Tips](#-animator--capture-best-practices)

</div>

---

## ⚡ Why 3D Artists & Animators Love vid2anim

Most AI mocap tools force you into proprietary cloud subscriptions, require expensive multi-thousand-dollar GPUs, or spit out locked skeletons that don't match your character's armature.

`vid2anim` is an **open-source, artist-friendly 3D workstation tool** designed to fit right into your existing Blender, Unreal, Unity, or VTuber pipeline:

* **🚀 100% CPU-Ready (No Expensive GPU Required):** Runs smoothly on standard laptops and workstations using optimized computer vision, with optional NVIDIA CUDA acceleration when you want ultra-fast batch processing.
* **🔌 Native Blender 3D Add-on:** Retarget reference video directly onto your character armature inside Blender with a single click. Bakes standard keyframes straight to the Blender Timeline & Action Editor.
* **🌐 Interactive Web Studio:** Drag-and-drop your video and 3D character (`.glb`, `.fbx`, `.obj`). Preview synchronized side-by-side motion in real time with an interactive Three.js 3D orbit viewport before exporting.
* **🖐️ 5-Digit Anatomical Finger Articulation:** Captures independent Thumb, Index, Middle, Ring, and Pinky flexion. No rigid "mitten hands" or static fists.
* **🎯 Universal Rig Auto-Detection:** Automatically matches bones for **Mixamo, VRM (Ready Player Me / VTubers), Blender Rigify, Unreal Engine Mannequins (UE4/UE5), and Daz3D / Character Creator**.
* **🦴 Auto-Rigging for Static Meshes:** Have an unrigged 3D statue or creature mesh? The built-in auto-rigger automatically fits a humanoid skeleton and computes smooth skinning weights.
* **📦 Universal glTF 2.0 & BVH Export:** Exports clean, strictly compliant `.glb` files ready for Blender, Unreal Engine, Unity, Three.js, Godot, and Web, plus standard `.bvh` files for Maya, Cinema 4D, and 3ds Max.

---

## 🎬 Three Ways to Use vid2anim

Choose the workflow that fits your creative setup:

### 1. In Blender (1-Click Viewport Add-on)
> **Best for:** 3D Animators, Riggers, and Character Artists.

No need to leave Blender! Install the included add-on (`plugins/blender/vid2anim_blender.py`):
1. Press <kbd>N</kbd> in the 3D Viewport to open the sidebar tab **`vid2anim`**.
2. Select your character's armature in the scene.
3. Choose your reference video or photo.
4. Click **"Animate Selected Armature"**.
5. Keyframes are baked directly into a native Blender **Action**, ready for tweaking in the Graph Editor or NLA Stacks.

---

### 2. In Your Browser (Web Studio & Real-Time 3D Player)
> **Best for:** Quick previews, game devs, VTubers, and visual inspection.

Launch the local studio with one command:
```bash
vid2anim serve
```
Open `http://localhost:8000` to access:
* **Drag-and-Drop Station:** Drop your reference `.mp4` / `.jpg` on the left, and your `.glb` / `.fbx` character on the right.
* **Synchronized Side-by-Side Playback:** Scrub video and 3D motion together with sub-frame timecode display and frame-stepping buttons (<kbd>◀</kbd> / <kbd>▶</kbd>).
* **Interactive 3D Orbit Viewport:** Rotate, pan, zoom, inspect skeleton bones, toggle wireframe, and check ground alignment.
* **1-Click Download:** Download the animated `.glb` or export universal `.bvh` motion files instantly.

---

### 3. Command Line & Automation (Pipeline TDs & Batch Work)
> **Best for:** Technical Directors, game studios, and processing entire motion libraries.

```bash
# Animate a character directly from video:
vid2anim run --input dance.mp4 --model character.glb --output animated.glb

# Extract motion once to cache, then retarget onto 10 different characters:
vid2anim extract --input stunts.mp4 --output motion.npz
vid2anim retarget --motion motion.npz --model warrior.glb --output warrior_anim.glb
vid2anim retarget --motion motion.npz --model robot.glb   --output robot_anim.glb
vid2anim retarget --motion motion.npz --model vtuber.vrm  --output vtuber_anim.glb
```

---

## 🦾 Supported Character Rigs & Formats

`vid2anim` comes pre-configured with bone mapping dictionaries and topological solvers for the world's most popular 3D character ecosystems:

| Character Rig / Ecosystem | Common Bone Prefixes | Finger Tracking | Default Rest Pose | Target Tools & Engines |
|:---|:---|:---:|:---:|:---|
| **Mixamo** | `mixamorig:Hips`, `mixamorig:LeftArm` | All 5 Digits | T-Pose | Blender, Unity, Unreal, Web |
| **VRM / VTuber** | `J_Bip_C_Hips`, `hips`, `leftUpperArm` | All 5 Digits | T-Pose / A-Pose | VRChat, VTubing, Unity, Three.js |
| **Ready Player Me** | `Armature/Hips`, `LeftHandIndex1` | All 5 Digits | T-Pose | Web3D, Unity, Unreal, Godot |
| **Blender Rigify** | `DEF-spine`, `DEF-upper_arm.L` | All 5 Digits | T-Pose / A-Pose | Blender Film & VFX Pipeline |
| **Unreal Engine Mannequin** | `pelvis`, `upperarm_l`, `hand_l` | All 5 Digits | A-Pose (30° / 45°) | Unreal Engine 4 & 5 Blueprints |
| **Daz3D / Character Creator** | `hip`, `lShldr`, `lForeArm` | All 5 Digits | T-Pose | Cinematics, Metahumans |
| **Static Meshes (Auto-Rig)** | *Raw unrigged mesh* | Procedural | Auto-Fitted | Any `.glb`, `.gltf`, or `.obj` |

> **Tip:** Not sure how your model is rigged? Run `vid2anim inspect your_model.glb` to get an instant terminal breakdown of all bones, detected rig profile, and joint bindings.

---

## 🎨 What Makes It Built for 3D Artists?

### 🖐️ Natural 5-Digit Hand Tracking
Most AI motion systems leave hands frozen or lock fingers into rigid blocks. `vid2anim` models natural human hand anatomy with independent flexion and curling cascades for:
* **Thumb:** CMC, MCP, and IP articulation.
* **Fingers:** Proximal, Intermediate, and Distal joints for Index, Middle, Ring, and Pinky.
* **Auto-Curling:** Captures expressive gestures, pointing, open palms, and clenched fists.

### 🧘 Zero Gimbal Lock & No 360° "Bone Pop" Flips
By using closed-form Lie algebra basis transformations ($C D C^T$) and continuous quaternion shortest-arc SLERP, joints never flip 180°/360° or suffer from Euler angle gimbal lock when limbs cross axes.

### 🌊 Adaptive Anti-Jitter Filtering
Raw computer vision mocap is notoriously noisy. `vid2anim` applies temporal smoothing with velocity spike clamping:
* Eliminates micro-shiver and jitter on stationary joints.
* Preserves sharp, high-velocity movements (punches, kicks, fast dance steps).

### 🦶 Root Grounding & Foot Planting
Calculates hip trajectory relative to ground contact, preventing characters from floating in mid-air or sinking into the floor during squats and lunges.

---

## 🚀 2-Minute Quickstart

### 1. Installation

Requires **Python 3.10, 3.11, or 3.12**:

```bash
# Clone the repository
git clone https://github.com/mehulcode12/vid2anim.git
cd vid2anim

# Create & activate a virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

# Install vid2anim (CPU mode by default)
pip install -e .
```

*(Optional: For NVIDIA GPU acceleration, run `pip install -e .[gpu]`)*

### 2. Verify Your System
```bash
vid2anim doctor
```
Checks your CPU cores, compute device, and MediaPipe AI tracking engine.

### 3. Launch the Web Studio
```bash
vid2anim serve
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser and start animating!

---

## 🧩 Blender 3D Bridge Add-on

`vid2anim` includes a dedicated bridge add-on for **Blender 3.0, 3.6 LTS, 4.0, 4.1, and 4.2+ LTS**.

### How to Install:
1. Start the local server: `vid2anim serve`
2. Open Blender &rarr; **Edit &rarr; Preferences &rarr; Add-ons**.
3. Click **Install...** (or the arrow dropdown in 4.2) and select:
   ```
   plugins/blender/vid2anim_blender.py
   ```
   *(Or click "Download Blender Add-on" from the Web Studio header).*
4. Enable the checkbox for **Animation: vid2anim Blender Bridge**.

### How to Use:
1. In the 3D Viewport, press <kbd>N</kbd> to toggle the sidebar, and click the **`vid2anim`** tab.
2. Select your character's armature.
3. Browse for your reference video or photo.
4. Click **"Animate Selected Armature"**.
5. Done! The animation is now a native Blender Action.

---

## 💡 Animator & Capture Best Practices

For the cleanest, production-ready mocap results:

| Factor | Recommended | Avoid |
|:---|:---|:---|
| **Framing** | Full body visible (head to toes in frame) | Cut-off feet or cropped hands |
| **Camera** | Static camera at chest height | Shaky handheld, extreme bird's-eye / worm's-eye |
| **Lighting** | Clear contrast between actor and background | Strong silhouettes or dark environments |
| **Clothing** | Form-fitting or casual everyday clothes | Baggy dresses, long cloaks, heavy trench coats |
| **Model Rest Pose** | Clean T-Pose or 45° A-Pose | Custom distorted bind poses |

---

## 💻 CLI Reference Cheat Sheet

| Command | Usage | Description |
|:---|:---|:---|
| `vid2anim serve` | `vid2anim serve --port 8000` | Launch local Web Studio & 3D player |
| `vid2anim inspect` | `vid2anim inspect character.glb` | Inspect bones, rig type, and rest pose |
| `vid2anim run` | `vid2anim run -i clip.mp4 -m rig.glb -o out.glb` | Full end-to-end animation |
| `vid2anim extract` | `vid2anim extract -i clip.mp4 -o motion.npz` | Extract motion to reusable cache |
| `vid2anim retarget` | `vid2anim retarget -m motion.npz -m rig.glb` | Retarget cached motion onto any model |
| `vid2anim doctor` | `vid2anim doctor` | Check hardware, CPU, and AI models |

---

## 🛠️ Python Developer API

Integrate retargeting into your custom Python DCC pipelines:

```python
from pathlib import Path
from vid2anim import PipelineConfig, Vid2AnimPipeline

config = PipelineConfig(
    input_media=Path("mocap_reference.mp4"),
    target_model=Path("game_character.glb"),
    output_glb=Path("output/game_character_animated.glb"),
    smooth=True,              # Anti-jitter filter
    max_duration_sec=60.0,
)

pipeline = Vid2AnimPipeline(config)
result = pipeline.execute()

if result.success:
    print(f"Animation saved: {result.output_glb} ({result.frame_count} frames)")
```

---

## 🤝 Contributing

We welcome contributions from animators, technical directors, and developers!
* Found a character rig that doesn't map cleanly? Open an issue with your bone hierarchy!
* Have ideas for Blender add-on UI improvements? Pull requests are warmly welcomed.

---

## 📄 License

This project is licensed under the **MIT License** — free for commercial, indie, and personal use.
