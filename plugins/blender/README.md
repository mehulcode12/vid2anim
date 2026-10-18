# vid2anim Blender Bridge Add-on

Direct 1-click video-to-animation retargeting inside Blender.

---

## Features
- **Native 3D Viewport Panel**: Lives in the Blender `N`-sidebar under the **`vid2anim`** tab.
- **1-Click Retargeting**: Select any rigged character/armature in your Blender scene, choose a reference video or photo, and click **"Animate Selected Armature"**.
- **Zero Third-Party Pip Dependencies in Blender**: Uses Blender's native Python standard library and glTF 2.0 importer/exporter. All AI computation runs cleanly in the `vid2anim` background server or GPU.
- **Armature Compatibility**: Works with **Mixamo**, **VRM / Ready Player Me**, **Blender Rigify**, **Unreal Engine Mannequin**, **Character Creator / Daz3D**, and custom humanoid rigs.
- **Automatic Timeline & Action Setup**: Bakes keyframes directly to a named Blender `Action` and optionally loops/plays in the viewport immediately.

---

## Installation (Blender 3.0 / 3.6 LTS / 4.0 / 4.1 / 4.2+)

1. In Blender, open **Edit &rarr; Preferences &rarr; Add-ons**.
2. Click **Install...** (top right).
3. Select `plugins/blender/vid2anim_blender.py` (or download it from the vid2anim Web UI).
4. Enable the checkbox for **Animation: vid2anim Blender Bridge**.

---

## How to Use

1. **Start the vid2anim Server**:
   In your terminal, start the local server:
   ```bash
   vid2anim serve --port 8080
   ```

2. **In Blender**:
   - Open the 3D Viewport and press <kbd>N</kbd> to reveal the sidebar.
   - Click the **`vid2anim`** tab.
   - Click **"Test Connection"** to verify server connectivity.
   - Select your character or armature in the 3D Viewport.
   - Under **Reference Media**, pick your video (`.mp4`, `.mov`, `.webm`) or photo.
   - Choose your engine (**MediaPipe Pose**, **DWPose WholeBody**, or **GVHMR Studio**) and device (**CPU** / **CUDA**).
   - Click **"Animate Selected Armature"**.

3. **Result**:
   The animation action is automatically bound to your active armature and ready for playback or editing in Blender's Dope Sheet / Non-Linear Animation (NLA) editor!
