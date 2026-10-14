"""vid2anim Blender Bridge Add-on.

Enables 1-click AI video-to-animation retargeting directly within Blender.
Communicates with the local vid2anim server (or CLI) to extract 3D motion from
any reference video or photo and apply it straight to your active armature.

Supported Blender Versions: 3.0.0, 3.6 LTS, 4.0, 4.1, 4.2+
Requires no third-party pip dependencies inside Blender.
"""

bl_info = {
    "name": "vid2anim Blender Bridge",
    "author": "Mehul & vid2anim contributors",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > vid2anim",
    "description": "Animate any 3D character armature directly from video or photo with AI motion capture.",
    "warning": "",
    "doc_url": "https://github.com/mehulcode12/vid2anim",
    "category": "Animation",
}

import email.generator
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty
    from bpy.types import Operator, Panel, PropertyGroup
except ImportError:
    # Allows headless unit testing outside Blender
    bpy = None


# ---------------------------------------------------------------------------
# Multipart HTTP Client (Standard Library only - zero external dependencies)
# ---------------------------------------------------------------------------

def _encode_multipart_form(fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    """Encode multipart/form-data payload using standard library.
    
    files dict format: key -> (filename, file_bytes, content_type)
    """
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    for key, (filename, data, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type_header


def check_server_status(server_url: str) -> Tuple[bool, str]:
    """Ping vid2anim server health check endpoint."""
    url = server_url.rstrip("/") + "/api/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vid2anim-Blender-Bridge/0.1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "ok":
                    return True, "Server Online (Ready)"
    except urllib.error.URLError:
        return False, "Server Offline (Run: vid2anim serve)"
    except Exception as e:
        return False, f"Connection Error: {e}"
    return False, "Server Offline"


# ---------------------------------------------------------------------------
# Blender Properties
# ---------------------------------------------------------------------------

if bpy:
    class Vid2AnimProperties(PropertyGroup):
        server_url: StringProperty(
            name="Server URL",
            description="Address of the running local vid2anim Web UI & API server",
            default="http://127.0.0.1:8000",
        )
        media_path: StringProperty(
            name="Media File",
            description="Select input reference video (.mp4, .mov, .webm) or photo (.jpg, .png)",
            subtype="FILE_PATH",
            default="",
        )
        engine: EnumProperty(
            name="Tracking Engine",
            description="AI motion tracking model to use for extraction",
            items=[
                ("mediapipe", "MediaPipe Pose", "Fast CPU-based tracking (~30 FPS)"),
                ("dwpose", "DWPose WholeBody", "WholeBody 133-point deep AI tracking"),
            ],
            default="mediapipe",
        )
        device: EnumProperty(
            name="Execution Device",
            description="Hardware execution device",
            items=[
                ("cpu", "CPU", "Execute on host CPU"),
                ("cuda", "CUDA (GPU)", "Accelerate on NVIDIA GPU"),
            ],
            default="cpu",
        )
        action_name: StringProperty(
            name="Action Name",
            description="Name for the newly created Blender Action",
            default="Vid2Anim_Motion",
        )
        auto_play: BoolProperty(
            name="Auto Play",
            description="Immediately start playback in Blender viewport after retargeting",
            default=True,
        )
        sync_timeline: BoolProperty(
            name="Set Timeline End",
            description="Automatically adjust scene frame_end to match animation duration",
            default=True,
        )


# ---------------------------------------------------------------------------
# Blender Operators
# ---------------------------------------------------------------------------

if bpy:
    class VID2ANIM_OT_CheckServer(Operator):
        """Test connection to the vid2anim local server."""
        bl_idname = "vid2anim.check_server"
        bl_label = "Test Server Connection"
        bl_description = "Check if the vid2anim server is running on the specified URL"

        def execute(self, context):
            props = context.scene.vid2anim_props
            is_online, msg = check_server_status(props.server_url)
            if is_online:
                self.report({'INFO'}, f"Connected: {msg}")
            else:
                self.report({'WARNING'}, f"{msg}. Start the server via terminal: vid2anim serve")
            return {'FINISHED'}


    class VID2ANIM_OT_OpenWebUI(Operator):
        """Open the vid2anim interactive 3D Web UI in browser."""
        bl_idname = "vid2anim.open_webui"
        bl_label = "Open 3D Web UI"
        bl_description = "Launch the interactive 3D viewport and inspection tool in your browser"

        def execute(self, context):
            props = context.scene.vid2anim_props
            webbrowser.open(props.server_url)
            self.report({'INFO'}, f"Opened {props.server_url}")
            return {'FINISHED'}


    class VID2ANIM_OT_AnimateArmature(Operator):
        """Export active armature, retarget motion from video, and apply action."""
        bl_idname = "vid2anim.animate_armature"
        bl_label = "Animate Selected Armature"
        bl_description = "Extract 3D motion from the media file and retarget directly onto the active armature"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            props = context.scene.vid2anim_props

            # 1. Validation: Verify active object is an Armature
            armature_obj = context.active_object
            if not armature_obj:
                self.report({'ERROR'}, "Please select an armature or rigged character.")
                return {'CANCELLED'}

            # If user selected a skinned mesh, find its parent armature
            if armature_obj.type != 'ARMATURE':
                parent_arm = armature_obj.find_armature() if hasattr(armature_obj, 'find_armature') else None
                if not parent_arm and armature_obj.parent and armature_obj.parent.type == 'ARMATURE':
                    parent_arm = armature_obj.parent
                if parent_arm:
                    armature_obj = parent_arm
                    context.view_layer.objects.active = armature_obj
                else:
                    self.report({'ERROR'}, f"Selected object '{armature_obj.name}' is not an Armature and has no parent armature.")
                    return {'CANCELLED'}

            # 2. Validation: Verify media file path
            media_path = bpy.path.abspath(props.media_path)
            if not media_path or not os.path.exists(media_path):
                self.report({'ERROR'}, f"Media file not found: {props.media_path}")
                return {'CANCELLED'}

            # 3. Check server liveness
            is_online, status_msg = check_server_status(props.server_url)
            if not is_online:
                self.report(
                    {'ERROR'},
                    f"vid2anim server is offline at {props.server_url}. Please run 'vid2anim serve' in terminal first."
                )
                return {'CANCELLED'}

            self.report({'INFO'}, f"Exporting '{armature_obj.name}' for AI motion retargeting...")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir = Path(tmpdir)
                export_glb_path = tmp_dir / "target_character.glb"
                result_glb_path = tmp_dir / "animated_character.glb"

                # 4. Export selected character as temporary GLB
                # Save current selection state
                prev_selected = list(context.selected_objects)
                prev_active = context.active_object

                try:
                    # Select armature and all its skinned children
                    bpy.ops.object.select_all(action='DESELECT')
                    armature_obj.select_set(True)
                    for child in armature_obj.children:
                        if child.type == 'MESH':
                            child.select_set(True)
                    context.view_layer.objects.active = armature_obj

                    # Export selected armature & mesh to temporary GLB
                    bpy.ops.export_scene.gltf(
                        filepath=str(export_glb_path),
                        use_selection=True,
                        export_format='GLB',
                        export_animations=False,
                        export_skins=True,
                        export_morph=True,
                    )
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to export character to glTF: {e}")
                    return {'CANCELLED'}
                finally:
                    # Restore previous selection
                    bpy.ops.object.select_all(action='DESELECT')
                    for o in prev_selected:
                        if o.name in bpy.data.objects:
                            o.select_set(True)
                    if prev_active and prev_active.name in bpy.data.objects:
                        context.view_layer.objects.active = prev_active

                if not export_glb_path.exists() or export_glb_path.stat().st_size == 0:
                    self.report({'ERROR'}, "glTF export generated an empty file.")
                    return {'CANCELLED'}

                # 5. POST to vid2anim server
                model_bytes = export_glb_path.read_bytes()
                media_bytes = Path(media_path).read_bytes()
                media_filename = Path(media_path).name

                content_type = "video/mp4"
                if media_filename.lower().endswith((".jpg", ".jpeg")):
                    content_type = "image/jpeg"
                elif media_filename.lower().endswith(".png"):
                    content_type = "image/png"
                elif media_filename.lower().endswith(".webm"):
                    content_type = "video/webm"

                fields = {
                    "device": props.device,
                    "engine": props.engine,
                }
                files = {
                    "model": ("character.glb", model_bytes, "model/gltf-binary"),
                    "media": (media_filename, media_bytes, content_type),
                }

                payload, ctype_hdr = _encode_multipart_form(fields, files)
                post_url = props.server_url.rstrip("/") + "/api/animate"

                self.report({'INFO'}, f"Sending to vid2anim ({props.engine.upper()} on {props.device.upper()})...")

                try:
                    req = urllib.request.Request(
                        post_url,
                        data=payload,
                        headers={
                            "Content-Type": ctype_hdr,
                            "User-Agent": "vid2anim-Blender-Bridge/0.1.0",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=300.0) as resp:
                        if resp.status != 200:
                            self.report({'ERROR'}, f"Pipeline failed with HTTP {resp.status}")
                            return {'CANCELLED'}
                        result_glb_path.write_bytes(resp.read())
                except urllib.error.HTTPError as e:
                    err_msg = e.read().decode("utf-8", errors="ignore")
                    self.report({'ERROR'}, f"Pipeline error: {err_msg or e.reason}")
                    return {'CANCELLED'}
                except Exception as e:
                    self.report({'ERROR'}, f"Request failed: {e}")
                    return {'CANCELLED'}

                # 6. Import animated GLB and extract Action
                before_actions = set(bpy.data.actions.keys())
                before_objects = set(bpy.data.objects.keys())

                try:
                    bpy.ops.import_scene.gltf(
                        filepath=str(result_glb_path),
                        import_pack_images=False,
                    )
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to import returned animation: {e}")
                    return {'CANCELLED'}

                after_actions = set(bpy.data.actions.keys())
                after_objects = set(bpy.data.objects.keys())

                new_actions = list(after_actions - before_actions)
                new_objects = [bpy.data.objects[name] for name in (after_objects - before_objects) if name in bpy.data.objects]

                if not new_actions:
                    self.report({'ERROR'}, "No new animation action found in retargeted asset.")
                    # Clean up imported objects
                    for o in new_objects:
                        bpy.data.objects.remove(o, do_unlink=True)
                    return {'CANCELLED'}

                # Prefer action named 'Vid2Anim_Motion' or take the first newly imported one
                target_action_name = next((a for a in new_actions if "Vid2Anim" in a), new_actions[0])
                target_action = bpy.data.actions[target_action_name]

                # Rename the action cleanly
                final_action_name = props.action_name or "Vid2Anim_Motion"
                target_action.name = final_action_name

                # Clean up temporary imported objects and meshes
                for o in new_objects:
                    if o.type == 'MESH' and o.data:
                        mesh_data = o.data
                        bpy.data.objects.remove(o, do_unlink=True)
                        if mesh_data.users == 0:
                            bpy.data.meshes.remove(mesh_data)
                    else:
                        bpy.data.objects.remove(o, do_unlink=True)

                # 7. Apply Action to active armature
                if not armature_obj.animation_data:
                    armature_obj.animation_data_create()
                armature_obj.animation_data.action = target_action

                # 8. Adjust scene timeline
                if props.sync_timeline and target_action.frame_range:
                    context.scene.frame_start = int(target_action.frame_range[0])
                    context.scene.frame_end = int(target_action.frame_range[1])
                    context.scene.frame_current = context.scene.frame_start

                if props.auto_play:
                    bpy.ops.screen.animation_play()

                total_frames = int(target_action.frame_range[1] - target_action.frame_range[0]) if target_action.frame_range else 0
                self.report({'INFO'}, f"Applied '{final_action_name}' ({total_frames} frames) to armature '{armature_obj.name}'!")
                return {'FINISHED'}


# ---------------------------------------------------------------------------
# Viewport Sidebar Panel (N-Panel)
# ---------------------------------------------------------------------------

if bpy:
    class VID2ANIM_PT_MainPanel(Panel):
        """vid2anim main 3D Viewport sidebar panel."""
        bl_label = "vid2anim Bridge"
        bl_idname = "VID2ANIM_PT_MainPanel"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = 'vid2anim'

        def draw(self, context):
            layout = self.layout
            props = context.scene.vid2anim_props

            # 1. Server Connection Status
            box = layout.box()
            row = box.row(align=True)
            row.label(text="Server:", icon='SERVER')
            row.prop(props, "server_url", text="")
            
            row_actions = box.row(align=True)
            row_actions.operator("vid2anim.check_server", text="Test Connection", icon='FILE_REFRESH')
            row_actions.operator("vid2anim.open_webui", text="Web UI", icon='URL')

            # 2. Source Media File Picker
            box_media = layout.box()
            box_media.label(text="Reference Media", icon='FILE_MOVIE')
            box_media.prop(props, "media_path", text="")

            # 3. Target Armature Status
            box_arm = layout.box()
            box_arm.label(text="Target Character", icon='ARMATURE_DATA')
            active_obj = context.active_object
            if active_obj:
                if active_obj.type == 'ARMATURE':
                    row_arm = box_arm.row()
                    row_arm.label(text=f"Selected: {active_obj.name}", icon='CHECKMARK')
                elif hasattr(active_obj, 'find_armature') and active_obj.find_armature():
                    row_arm = box_arm.row()
                    row_arm.label(text=f"Parent: {active_obj.find_armature().name}", icon='CHECKMARK')
                else:
                    box_arm.label(text="Select an armature in viewport", icon='ERROR')
            else:
                box_arm.label(text="No object selected", icon='INFO')

            # 4. Engine & Device Settings
            box_settings = layout.box()
            box_settings.label(text="Extraction Settings", icon='PREFERENCES')
            box_settings.prop(props, "engine")
            box_settings.prop(props, "device")

            # 5. Output Action Options
            box_opts = layout.box()
            box_opts.label(text="Animation Options", icon='ACTION')
            box_opts.prop(props, "action_name")
            row_opts = box_opts.row()
            row_opts.prop(props, "auto_play")
            row_opts.prop(props, "sync_timeline")

            # 6. Main Action Trigger Button
            layout.separator()
            btn_row = layout.row()
            btn_row.scale_y = 1.6
            btn_row.operator("vid2anim.animate_armature", text="Animate Selected Armature", icon='POSE_HLT')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    Vid2AnimProperties,
    VID2ANIM_OT_CheckServer,
    VID2ANIM_OT_OpenWebUI,
    VID2ANIM_OT_AnimateArmature,
    VID2ANIM_PT_MainPanel,
) if bpy else ()


def register():
    if bpy:
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.vid2anim_props = PointerProperty(type=Vid2AnimProperties)


def unregister():
    if bpy:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        if hasattr(bpy.types.Scene, "vid2anim_props"):
            del bpy.types.Scene.vid2anim_props


if __name__ == "__main__":
    register()
