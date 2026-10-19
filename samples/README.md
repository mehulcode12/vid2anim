# vid2anim Official Sample Assets

This folder contains ready-to-use sample 3D characters and reference video footage for quick testing, benchmarking, and demonstration of `vid2anim`.

> [!TIP]
> **See these sample assets in action:**
> - [▶ Watch Synchronized Multi-Character Demo on YouTube](https://youtu.be/Y3eJLQgHwsc)
> - [▶ Watch Full Web Studio Walkthrough on YouTube](https://youtu.be/4eaRWq1vsGs)

---

## 📦 Contents

### 1. 3D Character Models (`samples/models/`)

| Model | Format | Rig Type | Bones | Fingers | Size | Best For |
|:---|:---:|:---|:---:|:---:|:---:|:---|
| **`ybot.glb`** | `.glb` (glTF 2.0) | Adobe Mixamo Humanoid | 52 | Standard | 2.2 MB | Lightweight quick tests, fast CPU inference, CI testing |
| **`eve.glb`** | `.glb` (glTF 2.0) | Adobe Mixamo Humanoid | 65 | Full 5-Digit Anatomical | 18.3 MB | High-detail textured character, expressive finger gesture capture |

*Both models are standard glTF 2.0 binary files compatible with Blender, Unreal Engine, Unity, Three.js, Godot, and Web3D.*

---

### 2. Reference Footage (`samples/videos/`)

| Video | Resolution | FPS | Duration | Size | Movement Style |
|:---|:---:|:---:|:---:|:---:|:---|
| **`dance_urban.mp4`** | 1080×1920 (9:16) | 25 FPS | 11.2s | 6.8 MB | Urban rhythm choreography with sharp arm poses and foot grounding |
| **`dance_street.mp4`** | 1080×1920 (9:16) | 25 FPS | 15.2s | 10.3 MB | Street dance freestyle with dynamic torso, leg travel, and hand gestures |

---

## 🚀 Quick Testing Commands

### A. Quick 100% CPU Motion Capture (MediaPipe)
Runs on any standard PC or laptop in seconds:
```bash
vid2anim run \
  --input samples/videos/dance_urban.mp4 \
  --model samples/models/ybot.glb \
  --engine mediapipe \
  --output outputs/ybot_urban_dance.glb
```

### B. High-Fidelity 5-Digit Finger Articulation (DWPose)
Captures expressive finger curling and gestures on Eve:
```bash
vid2anim run \
  --input samples/videos/dance_urban.mp4 \
  --model samples/models/eve.glb \
  --engine dwpose \
  --output outputs/eve_finger_dance.glb
```

### C. Extract Motion Once, Retarget to Multiple Characters
```bash
# 1. Extract motion data to reusable numpy archive
vid2anim extract --input samples/videos/dance_street.mp4 --engine mediapipe --output outputs/street_motion.npz

# 2. Retarget onto both sample characters
vid2anim retarget --motion outputs/street_motion.npz --model samples/models/ybot.glb --output outputs/ybot_street.glb
vid2anim retarget --motion outputs/street_motion.npz --model samples/models/eve.glb  --output outputs/eve_street.glb
```

### D. Interactive Web Studio Testing
1. Start the local server:
   ```bash
   vid2anim serve --port 8080
   ```
2. Open `http://localhost:8080` in your browser.
3. Drag `samples/videos/dance_urban.mp4` into the Video dropzone.
4. Drag `samples/models/ybot.glb` into the Character dropzone.
5. Click **"Animate Character"** to preview side-by-side synchronized motion!

---

## 📜 Credits & License
- **Characters:** Created via Adobe Mixamo. Free to use for personal and commercial 3D animation projects.
- **Reference Footage:** Sourced from royalty-free open creative libraries (Pexels / Creative Commons Free to Use).
