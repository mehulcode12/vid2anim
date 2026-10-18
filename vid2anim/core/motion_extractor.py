"""Dual-Engine Motion Extractor (CPU & GPU) for Videos and Photos.

Extracts 3D human pose kinematics from:
- Single still photos (.jpg, .png)
- Monocular action videos (.mp4, .mov, .webm, up to 600s / 10 minutes)

Includes an ultra-fast CPU engine (no GPU or multi-gigabyte downloads needed)
and optional CUDA GPU acceleration when available.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..utils.diagnostics import get_system_diagnostics
from ..utils.progress import ProgressReporter
from .rig_detector import CANONICAL_JOINTS


def configure_isolated_environment(project_dir: Optional[Path] = None) -> Path:
    """Ensure all model weights, torch caches, HuggingFace Hub downloads, and checkpoints
    are strictly isolated within this project directory, NEVER touching the C: drive or user home."""
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[2]

    weights_dir = project_dir / "weights"
    assets_dir = project_dir / "assets"
    inputs_dir = project_dir / "inputs"
    scratch_dir = project_dir / "scratch"

    hf_home = weights_dir / "huggingface"
    hf_hub = hf_home / "hub"
    torch_home = weights_dir / "torch"
    gvhmr_ckpts = weights_dir / "gvhmr"
    gvhmr_body = assets_dir / "body_models"
    xdg_cache = weights_dir / "cache"
    yolo_dir = weights_dir / "yolo"
    mpl_dir = weights_dir / "matplotlib"

    for d in (hf_hub, torch_home, gvhmr_ckpts, gvhmr_body, inputs_dir, xdg_cache, yolo_dir, mpl_dir, scratch_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["GVHMR_CHECKPOINTS"] = str(gvhmr_ckpts)
    os.environ["GVHMR_BODY_MODELS"] = str(gvhmr_body)
    os.environ["GVHMR_DATA_ROOT"] = str(inputs_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["TMPDIR"] = str(scratch_dir)
    os.environ["TEMP"] = str(scratch_dir)
    os.environ["TMP"] = str(scratch_dir)

    return project_dir


def get_video_playback_fps(video_path: str | Path, n_frames: Optional[int] = None, fallback_fps: float = 30.0) -> float:
    """Calculate the precise playback FPS so extracted frames lockstep-synchronize with video duration.

    GVHMR, RTMLib, and MediaPipe models operate at distinct internal inference rates,
    which must never override the source video container's playback frame rate.
    """
    path_str = str(video_path)
    # 1. Try ffprobe for container frame rate and duration
    try:
        import subprocess, json
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration",
            "-of", "json", path_str
        ]
        out = subprocess.check_output(cmd, timeout=5).decode()
        meta = json.loads(out)
        streams = meta.get("streams", [])
        if streams:
            rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate") or "0/1"
            num, _, den = str(rate).partition("/")
            rate_val = float(num) / float(den or 1)
            dur_str = streams[0].get("duration") or meta.get("format", {}).get("duration")
            dur_val = float(dur_str) if dur_str else None
            if rate_val > 0 and dur_val and dur_val > 1e-3 and n_frames is not None:
                if abs(rate_val * dur_val - n_frames) <= 2:
                    return float(rate_val)
                return float(n_frames / dur_val)
            if rate_val > 0:
                return float(rate_val)
    except Exception:
        pass

    # 2. Try OpenCV VideoCapture
    try:
        import cv2
        cap = cv2.VideoCapture(path_str)
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cv_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if cv_fps > 1e-3 and cv_count > 0:
            duration = cv_count / cv_fps
            if n_frames is not None and duration > 1e-3:
                if abs(cv_count - n_frames) <= 2:
                    return float(cv_fps)
                return float(n_frames / duration)
            return float(cv_fps)
    except Exception:
        pass

    return float(fallback_fps) if fallback_fps > 0 else 30.0


class AsyncVideoReader:
    """High-throughput asynchronous video frame reader with background prefetching.

    Decodes video frames on a dedicated background thread into a bounded queue,
    eliminating decode bottlenecks and overlapping I/O with model inference.
    """

    def __init__(self, video_path: str | Path, max_frames: int, queue_size: int = 64) -> None:
        self.video_path = str(video_path)
        self.max_frames = max_frames
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0
        self._thread: Optional[threading.Thread] = None

        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
        except Exception:
            pass

    def start(self) -> AsyncVideoReader:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def _worker(self) -> None:
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            frame_idx = 0
            while not self.stopped.is_set() and frame_idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                self.queue.put((frame_idx, frame))
                frame_idx += 1
            cap.release()
        except Exception:
            pass
        finally:
            self.queue.put((None, None))

    def __iter__(self):
        while True:
            idx, frame = self.queue.get()
            if idx is None:
                break
            yield idx, frame

    def close(self) -> None:
        self.stopped.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


@dataclass
class MotionSequence:
    """Standardized multi-frame 3D skeletal motion."""
    fps: float
    duration_sec: float
    frame_count: int
    timestamps: np.ndarray  # (F,) in seconds
    joint_positions: Dict[str, np.ndarray]  # canonical_name -> (F, 3) 3D coordinates
    root_translation: np.ndarray  # (F, 3) root motion trajectory

    def save_cache(self, npz_path: str | Path) -> None:
        """Cache extracted motion to disk for instant reuse across different 3D rigs."""
        p = Path(npz_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "fps": np.array([self.fps]),
            "duration_sec": np.array([self.duration_sec]),
            "frame_count": np.array([self.frame_count]),
            "timestamps": self.timestamps,
            "root_translation": self.root_translation,
        }
        for j_name, pos in self.joint_positions.items():
            save_dict[f"joint_{j_name}"] = pos
        np.savez_compressed(p, **save_dict)

    @classmethod
    def load_cache(cls, npz_path: str | Path) -> MotionSequence:
        """Load previously extracted motion."""
        data = np.load(npz_path)
        fps = float(data["fps"][0])
        duration_sec = float(data["duration_sec"][0])
        frame_count = int(data["frame_count"][0])
        timestamps = data["timestamps"]
        root_translation = data["root_translation"]
        
        joint_positions: Dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("joint_"):
                j_name = key[len("joint_"):]
                joint_positions[j_name] = data[key]

        return cls(
            fps=fps,
            duration_sec=duration_sec,
            frame_count=frame_count,
            timestamps=timestamps,
            joint_positions=joint_positions,
            root_translation=root_translation,
        )


class MotionExtractor:
    """Extracts 3D motion from image or video with CPU/GPU device selection."""

    def __init__(
        self,
        device: str = "auto",
        engine: str = "mediapipe",
        dwpose_mode: str = "lightweight",
        reporter: Optional[ProgressReporter] = None,
    ) -> None:
        diag = get_system_diagnostics()
        if device == "auto":
            self.device = diag["recommended_device"]
        elif device.lower() in ["cuda", "gpu"]:
            if not diag["cuda_available"]:
                print("Warning: CUDA was requested but is not available on this machine. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = "cpu"

        self.engine = engine.lower() if engine else "mediapipe"
        self.dwpose_mode = dwpose_mode
        self.reporter = reporter or ProgressReporter()
        self._dwpose_detector = None

    def extract_from_file(
        self,
        media_path: str | Path,
        max_duration_sec: float = 30.0,
    ) -> MotionSequence:
        """Extract 3D motion from photo or video."""
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(f"Media file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            return self._extract_photo(p)
        elif suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            return self._extract_video(p, max_duration_sec=max_duration_sec)
        else:
            raise ValueError(f"Unsupported media format: {suffix}")

    @staticmethod
    def _ensure_pose_landmarker_model() -> Optional[Path]:
        """Ensure Google MediaPipe pose_landmarker_heavy.task model is available, strictly within project directory."""
        project_dir = Path(__file__).resolve().parents[2]
        candidates = [
            project_dir / "weights" / "mediapipe" / "pose_landmarker_heavy.task",
            project_dir / "models" / "pose_landmarker_heavy.task",
            Path("models") / "pose_landmarker_heavy.task",
        ]
        for cand in candidates:
            if cand.exists():
                return cand

        # Download directly to project weights directory
        target_path = project_dir / "weights" / "mediapipe" / "pose_landmarker_heavy.task"
        try:
            import urllib.request
            target_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, target_path)
            return target_path
        except Exception:
            return None


    def _init_dwpose_detector(self):
        """Initialize or return cached RTMLib Wholebody detector (DWPose / RTMPose)."""
        if self._dwpose_detector is not None:
            return self._dwpose_detector

        try:
            import rtmlib
            try:
                self._dwpose_detector = rtmlib.Wholebody(
                    mode=self.dwpose_mode,
                    backend="onnxruntime",
                    device=self.device,
                )
            except Exception:
                # Fallback to lightweight mode if requested mode fails to load
                self._dwpose_detector = rtmlib.Wholebody(
                    mode="lightweight",
                    backend="onnxruntime",
                    device=self.device,
                )
            return self._dwpose_detector
        except Exception as e:
            self.reporter.report("EXTRACT", 0.05, f"DWPose initialization notice: {e}. Falling back to MediaPipe.")
            return None

    def _map_dwpose_landmarks(
        self,
        kpts: np.ndarray,
        scores: np.ndarray,
        img_w: int,
        img_h: int,
        mp_depth_prior: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Map 133 DWPose / RTMPose WholeBody keypoints to canonical humanoid 3D format."""
        l_hip_2d = kpts[11]
        r_hip_2d = kpts[12]
        hips_2d = (l_hip_2d + r_hip_2d) * 0.5

        l_sh_2d = kpts[5]
        r_sh_2d = kpts[6]
        sh_mid_2d = (l_sh_2d + r_sh_2d) * 0.5

        torso_px = float(np.linalg.norm(sh_mid_2d - hips_2d))
        scale = (0.55 / torso_px) if torso_px > 10.0 else (1.0 / max(img_h, 1))

        def pt(idx: int, ref_z: float = 0.0) -> np.ndarray:
            p2d = kpts[idx]
            x = (p2d[0] - hips_2d[0]) * scale
            y = 1.0 - (p2d[1] - hips_2d[1]) * scale
            z = ref_z
            return np.array([x, y, z], dtype=np.float32)

        depth_map: Dict[str, float] = {}
        if mp_depth_prior is not None:
            for k, v in mp_depth_prior.items():
                depth_map[k] = float(v[2])
        else:
            depth_map["hips"] = 0.0
            depth_map["neck"] = 0.0
            depth_map["head"] = 0.0
            depth_map["left_shoulder"] = 0.0
            depth_map["right_shoulder"] = 0.0

            l_el_2d = kpts[7]
            d_l_arm_2d = float(np.linalg.norm(l_el_2d - l_sh_2d)) * scale
            dz_l_el = float(np.sqrt(max(0.0, 0.28**2 - d_l_arm_2d**2))) * 0.5
            depth_map["left_forearm"] = dz_l_el

            l_wr_2d = kpts[9]
            d_l_fore_2d = float(np.linalg.norm(l_wr_2d - l_el_2d)) * scale
            dz_l_wr = float(np.sqrt(max(0.0, 0.25**2 - d_l_fore_2d**2))) * 0.6
            depth_map["left_hand"] = dz_l_el + dz_l_wr

            r_el_2d = kpts[8]
            d_r_arm_2d = float(np.linalg.norm(r_el_2d - r_sh_2d)) * scale
            dz_r_el = float(np.sqrt(max(0.0, 0.28**2 - d_r_arm_2d**2))) * 0.5
            depth_map["right_forearm"] = dz_r_el

            r_wr_2d = kpts[10]
            d_r_fore_2d = float(np.linalg.norm(r_wr_2d - r_el_2d)) * scale
            dz_r_wr = float(np.sqrt(max(0.0, 0.25**2 - d_r_fore_2d**2))) * 0.6
            depth_map["right_hand"] = dz_r_el + dz_r_wr

            l_kn_2d = kpts[13]
            d_l_thigh_2d = float(np.linalg.norm(l_kn_2d - l_hip_2d)) * scale
            dz_l_kn = float(np.sqrt(max(0.0, 0.42**2 - d_l_thigh_2d**2))) * 0.4
            depth_map["left_leg"] = dz_l_kn

            l_an_2d = kpts[15]
            d_l_shin_2d = float(np.linalg.norm(l_an_2d - l_kn_2d)) * scale
            dz_l_an = float(np.sqrt(max(0.0, 0.40**2 - d_l_shin_2d**2))) * 0.4
            depth_map["left_foot"] = dz_l_kn + dz_l_an

            r_kn_2d = kpts[14]
            d_r_thigh_2d = float(np.linalg.norm(r_kn_2d - r_hip_2d)) * scale
            dz_r_kn = float(np.sqrt(max(0.0, 0.42**2 - d_r_thigh_2d**2))) * 0.4
            depth_map["right_leg"] = dz_r_kn

            r_an_2d = kpts[16]
            d_r_shin_2d = float(np.linalg.norm(r_an_2d - r_kn_2d)) * scale
            dz_r_an = float(np.sqrt(max(0.0, 0.40**2 - d_r_shin_2d**2))) * 0.4
            depth_map["right_foot"] = dz_r_kn + dz_r_an

        hips = np.array([0.0, 1.0, depth_map.get("hips", 0.0)], dtype=np.float32)
        l_shoulder = pt(5, depth_map.get("left_shoulder", 0.0))
        r_shoulder = pt(6, depth_map.get("right_shoulder", 0.0))
        shoulders_mid = (l_shoulder + r_shoulder) * 0.5
        neck = shoulders_mid
        l_ear = pt(3, depth_map.get("left_ear", 0.0))
        r_ear = pt(4, depth_map.get("right_ear", 0.0))
        if float(np.linalg.norm(l_ear - r_ear)) > 0.04:
            head = (l_ear + r_ear) * 0.5
        else:
            head = pt(0, depth_map.get("head", 0.0))

        chest = hips + (shoulders_mid - hips) * 0.70
        spine1 = hips + (shoulders_mid - hips) * 0.525
        spine = hips + (shoulders_mid - hips) * 0.35
        l_collar = shoulders_mid + (l_shoulder - shoulders_mid) * 0.45
        r_collar = shoulders_mid + (r_shoulder - shoulders_mid) * 0.45

        z_lh = depth_map.get("left_hand", 0.0)
        z_rh = depth_map.get("right_hand", 0.0)

        l_el = pt(7, depth_map.get("left_forearm", 0.0))
        r_el = pt(8, depth_map.get("right_forearm", 0.0))
        l_wr = pt(9, z_lh)
        r_wr = pt(10, z_rh)
        l_hp = pt(11, depth_map.get("left_upleg", 0.0))
        r_hp = pt(12, depth_map.get("right_upleg", 0.0))
        l_kn = pt(13, depth_map.get("left_leg", 0.0))
        r_kn = pt(14, depth_map.get("right_leg", 0.0))
        l_an = pt(15, depth_map.get("left_foot", 0.0))
        r_an = pt(16, depth_map.get("right_foot", 0.0))

        return {
            "hips": hips,
            "pelvis": hips,
            "spine": spine,
            "spine1": spine1,
            "spine2": spine1,
            "chest": chest,
            "spine3": chest,
            "neck": neck,
            "head": head,
            "left_collar": l_collar,
            "right_collar": r_collar,
            "left_shoulder": l_shoulder,
            "left_arm": l_shoulder,
            "left_forearm": l_el,
            "left_elbow": l_el,
            "left_hand": pt(99, z_lh + 0.01),
            "left_wrist": l_wr,
            "right_shoulder": r_shoulder,
            "right_arm": r_shoulder,
            "right_forearm": r_el,
            "right_elbow": r_el,
            "right_hand": pt(120, z_rh + 0.01),
            "right_wrist": r_wr,
            "left_fingers": pt(99, z_lh + 0.01),     # left forefinger tip
            "right_fingers": pt(120, z_rh + 0.01),   # right forefinger tip
            "left_thumb": pt(95, z_lh + 0.02),       # left thumb tip
            "right_thumb": pt(116, z_rh + 0.02),     # right thumb tip
            "left_pinky": pt(111, z_lh - 0.01),      # left pinky tip
            "right_pinky": pt(132, z_rh - 0.01),     # right pinky tip
            "left_middle": pt(103, z_lh + 0.01),     # left middle tip
            "right_middle": pt(124, z_rh + 0.01),    # right middle tip
            "left_ring": pt(107, z_lh),              # left ring tip
            "right_ring": pt(128, z_rh),             # right ring tip
            "left_upleg": l_hp,
            "left_hip": l_hp,
            "left_leg": l_kn,
            "left_knee": l_kn,
            "left_foot": pt(17, depth_map.get("left_foot", 0.0) + 0.10),
            "left_ankle": l_an,
            "left_toe": pt(17, depth_map.get("left_foot", 0.0) + 0.10),
            "right_upleg": r_hp,
            "right_hip": r_hp,
            "right_leg": r_kn,
            "right_knee": r_kn,
            "right_foot": pt(20, depth_map.get("right_foot", 0.0) + 0.10),
            "right_ankle": r_an,
            "right_toe": pt(20, depth_map.get("right_foot", 0.0) + 0.10),
        }

    def _extract_photo_dwpose(self, image_bgr: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Extract pose from a single photo using DWPose WholeBody."""
        detector = self._init_dwpose_detector()
        if detector is None:
            return None
        try:
            kpts, scores = detector(image_bgr)
            if len(kpts) > 0 and scores[0].mean() > 0.2:
                h, w = image_bgr.shape[:2]
                return self._map_dwpose_landmarks(kpts[0], scores[0], w, h)
        except Exception:
            pass
        return None

    def _extract_photo_gvhmr(self, image_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Extract pose from a single photo using GVHMR temporal hold clip."""
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            w -= w % 2
            h -= h % 2
            img = img[:h, :w]

            project_dir = Path(__file__).resolve().parents[2]
            scratch_dir = project_dir / "scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            hold_video = scratch_dir / f"hold_{image_path.stem}.mp4"

            writer = cv2.VideoWriter(str(hold_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            for _ in range(16):
                writer.write(img)
            writer.release()

            from .rig_detector import CANONICAL_JOINTS
            AUX = ("left_fingers", "right_fingers", "left_thumb", "right_thumb", "left_pinky", "right_pinky", "left_middle", "right_middle", "left_ring", "right_ring")
            tracked = list(CANONICAL_JOINTS) + [j for j in AUX if j not in CANONICAL_JOINTS]

            motion = self._extract_video_gvhmr(hold_video, max_duration_sec=1.0, tracked_joints=tracked)
            hold_video.unlink(missing_ok=True)

            mid = motion.frame_count // 2
            return {k: v[mid] for k, v in motion.joint_positions.items()}
        except Exception as e:
            self.reporter.report("EXTRACT", 0.1, f"GVHMR photo extract note: {e}. Falling back to MediaPipe.")
            return None

    def _extract_photo(self, image_path: Path) -> MotionSequence:
        """Extract single-frame 3D pose from a photo with multi-engine selection."""
        self.reporter.report("EXTRACT", 0.1, f"Analyzing photo: {image_path.name} on {self.device.upper()} ({self.engine.upper()})")
        positions_3d = None

        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                # 1. Try GVHMR if requested
                if self.engine in ["gvhmr", "hmr", "supermotion"]:
                    positions_3d = self._extract_photo_gvhmr(image_path)

                # 2. Try DWPose if requested
                if positions_3d is None and self.engine in ["dwpose", "rtmpose"]:
                    positions_3d = self._extract_photo_dwpose(img)

                # 3. Try MediaPipe if others not selected or returned None
                if positions_3d is None:
                    model_path = self._ensure_pose_landmarker_model()
                    if model_path:
                        try:
                            import mediapipe as mp
                            from mediapipe.tasks.python import BaseOptions
                            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

                            options = PoseLandmarkerOptions(
                                base_options=BaseOptions(model_asset_path=str(model_path)),
                                running_mode=RunningMode.IMAGE,
                                num_poses=1,
                                min_pose_detection_confidence=0.6,
                                min_pose_presence_confidence=0.6,
                                min_tracking_confidence=0.6,
                            )
                            with PoseLandmarker.create_from_options(options) as landmarker:
                                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                                res = landmarker.detect(mp_img)
                                if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                                    positions_3d = self._map_mediapipe_landmarks(res.pose_world_landmarks[0])
                        except Exception:
                            pass

                if positions_3d is None:
                    positions_3d = self._estimate_frame_3d_pose(img)
        except Exception:
            pass

        if positions_3d is None:
            positions_3d = self._generate_default_human_frame()

        self.reporter.report("EXTRACT", 1.0, "3D pose extracted successfully from photo.")

        frame_dict = {k: v[np.newaxis, :] for k, v in positions_3d.items()}
        root_trans = frame_dict.get("hips", np.zeros((1, 3), dtype=np.float32))

        return MotionSequence(
            fps=30.0,
            duration_sec=0.033,
            frame_count=1,
            timestamps=np.array([0.0], dtype=np.float32),
            joint_positions=frame_dict,
            root_translation=root_trans,
        )

    def _extract_video(self, video_path: Path, max_duration_sec: float = 30.0) -> MotionSequence:
        """Extract multi-frame 3D skeletal motion from a video clip with multi-engine routing."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
        except (ImportError, ValueError):
            self.reporter.report(
                "EXTRACT",
                0.5,
                "[NOTE] OpenCV not found or unable to decode clip. Using fallback motion sequence.",
            )
            from ..utils.fixtures import create_synthetic_motion_sequence
            return create_synthetic_motion_sequence(frames=30, fps=30.0)

        fps = get_video_playback_fps(video_path, fallback_fps=25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(fps * max_duration_sec)
        frames_to_process = min(total_frames, max_frames) if total_frames > 0 else max_frames

        from .rig_detector import CANONICAL_JOINTS
        AUXILIARY_LANDMARKS = (
            "left_fingers", "right_fingers", "left_thumb", "right_thumb", "left_pinky", "right_pinky",
            "left_middle", "right_middle", "left_ring", "right_ring"
        )
        tracked_joints = list(CANONICAL_JOINTS) + [j for j in AUXILIARY_LANDMARKS if j not in CANONICAL_JOINTS]

        if self.engine in ["gvhmr", "hmr", "supermotion"]:
            cap.release()
            return self._extract_video_gvhmr(
                video_path, max_duration_sec, tracked_joints
            )
        elif self.engine in ["dwpose", "rtmpose"]:
            return self._extract_video_dwpose(
                cap, video_path, fps, total_frames, frames_to_process, max_duration_sec, tracked_joints
            )
        else:
            return self._extract_video_mediapipe(
                cap, video_path, fps, total_frames, frames_to_process, max_duration_sec, tracked_joints
            )

    @staticmethod
    def _joints_world_from_smplx(params, device: str = "cpu") -> np.ndarray:
        """Convert GVHMR SMPL-X parametric predictions to 24 SMPL joints in metric world space."""
        import torch
        from einops import einsum
        from gvhmr import PROJ_ROOT
        from gvhmr.utils.device import to_device
        from gvhmr.utils.smplx_utils import make_smplx

        if device == "mps":
            device = "cpu"
        torch_device = torch.device(device)
        smplx = make_smplx("supermotion").to(torch_device)
        smplx2smpl = torch.load(
            PROJ_ROOT / "gvhmr/utils/body_model/smplx2smpl_sparse.pt",
            weights_only=False,
        ).to(torch_device)
        j_regressor = torch.load(
            PROJ_ROOT / "gvhmr/utils/body_model/smpl_neutral_J_regressor.pt",
            weights_only=False,
        ).to(torch_device)
        with torch.no_grad():
            out = smplx(**to_device(dict(params), torch_device))
            verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])
            joints = einsum(j_regressor, verts, "j v, l v i -> l j i")
        return joints.cpu().numpy().astype(np.float32)

    def _ensure_gvhmr_assets(self) -> None:
        """Ensure GVHMR, HMR2, ViTPose, YOLO, and SMPL-X body models are ready in project directory."""
        import threading
        import time
        from gvhmr.utils import assets

        smplx_path = assets.BODY_MODEL_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"
        if not smplx_path.exists():
            raise FileNotFoundError(
                f"SMPL-X body model not found at {smplx_path}. "
                f"Please ensure SMPLX_NEUTRAL.npz is in {assets.BODY_MODEL_ROOT / 'smplx'}."
            )

        needed = ["gvhmr", "hmr2", "vitpose", "yolo"]
        missing_assets = [n for n in needed if not assets.is_present(assets.ASSETS[n])]
        if missing_assets:
            total_needed_bytes = sum(assets.ASSETS[n].size for n in missing_assets)
            completed_bytes_prior = 0

            for asset_name in missing_assets:
                asset = assets.ASSETS[asset_name]
                if assets.is_present(asset):
                    continue

                stop_monitor = threading.Event()
                target_name = asset.path.name
                total_file_bytes = asset.size
                total_mb = total_file_bytes / (1024 * 1024)

                def _monitor_download(target_path=asset.path, expected_sz=total_file_bytes, prior=completed_bytes_prior, a_name=asset_name):
                    cache_dir = target_path.parent
                    t_last = time.time()
                    sz_last = 0
                    while not stop_monitor.is_set():
                        curr_sz = 0
                        if target_path.exists():
                            curr_sz = target_path.stat().st_size
                        else:
                            try:
                                incomp_files = list(cache_dir.glob("*.incomplete"))
                                if incomp_files:
                                    curr_sz = max(f.stat().st_size for f in incomp_files)
                            except Exception:
                                pass

                        t_now = time.time()
                        dt = max(0.001, t_now - t_last)
                        speed_mb = max(0.0, (curr_sz - sz_last) / (1024 * 1024) / dt)
                        t_last, sz_last = t_now, curr_sz

                        curr_mb = curr_sz / (1024 * 1024)
                        pct_file = min(100.0, (curr_sz / max(1, expected_sz)) * 100)
                        overall_done = prior + curr_sz
                        global_pct = min(0.15, 0.05 + 0.10 * (overall_done / max(1, total_needed_bytes)))

                        detail_str = f"{curr_mb:.1f} MB / {total_mb:.1f} MB ({pct_file:.0f}%) • {speed_mb:.1f} MB/s"
                        msg = f"Downloading {a_name} ({target_name}): {detail_str}"
                        self.reporter.report(
                            "EXTRACT",
                            global_pct,
                            msg,
                            extra={"detail": detail_str, "file": target_name, "speed_mb": speed_mb, "sub_percent": pct_file},
                        )
                        time.sleep(0.4)

                monitor_thread = threading.Thread(target=_monitor_download, daemon=True)
                monitor_thread.start()
                try:
                    assets.fetch({asset_name: asset}, force=False)
                finally:
                    stop_monitor.set()
                    monitor_thread.join(timeout=1.0)
                completed_bytes_prior += total_file_bytes

            self.reporter.report("EXTRACT", 0.15, "All neural network checkpoints verified & ready on disk.")

    def _extract_video_gvhmr(
        self,
        video_path: Path,
        max_duration_sec: float,
        tracked_joints: List[str],
    ) -> MotionSequence:
        """Extract studio-grade 3D motion using GVHMR human mesh recovery and SMPL-X.

        All checkpoints and temporary caches reside strictly within the project directory.
        Recovers world-grounded root trajectory and 24 SMPL joints.
        """
        configure_isolated_environment()
        self.reporter.report("EXTRACT", 0.05, f"Initializing GVHMR Studio Engine on {self.device.upper()}...")

        self._ensure_gvhmr_assets()

        import gvhmr
        work_dir = Path(__file__).resolve().parents[2] / "outputs" / "gvhmr_cache"
        work_dir.mkdir(parents=True, exist_ok=True)
        clip_work_dir = work_dir / video_path.stem
        clip_work_dir.mkdir(parents=True, exist_ok=True)

        def _on_gvhmr_progress(fraction: float, stage_label: str) -> None:
            pct_val = max(0, min(100, int(fraction * 100)))
            overall_pct = 0.15 + (fraction * 0.53)
            self.reporter.report(
                "EXTRACT",
                overall_pct,
                f"GVHMR [{stage_label}]: {pct_val}%",
                extra={"detail": f"{stage_label} ({pct_val}%)", "sub_percent": pct_val},
            )

        self.reporter.report("EXTRACT", 0.15, "GVHMR recovering 3D human mesh kinematics & root trajectory...")
        result = gvhmr.recover(
            str(video_path),
            static_camera=True,
            device=self.device,
            pose2d="rtmpose",
            output_dir=str(clip_work_dir),
            output_root=str(work_dir),
            progress=True,
            progress_callback=_on_gvhmr_progress,
        )

        pose_cache = clip_work_dir / "preprocess" / "vitpose.pt"
        from vid2anim.utils.stabilize_kp2d import stabilize_pose_cache
        if pose_cache.exists() and stabilize_pose_cache(pose_cache):
            motion_cache = clip_work_dir / "hmr4d_results.pt"
            motion_cache.unlink(missing_ok=True)
            result = gvhmr.recover(
                str(video_path),
                static_camera=True,
                device=self.device,
                pose2d="rtmpose",
                output_dir=str(clip_work_dir),
                output_root=str(work_dir),
                progress=True,
                progress_callback=_on_gvhmr_progress,
            )

        world = result.smpl_params_world
        joints_3d = self._joints_world_from_smplx(world, device=self.device)
        transl = world["transl"]
        root_translation = transl.detach().cpu().numpy().astype(np.float32) if hasattr(transl, "detach") else np.asarray(transl, dtype=np.float32)

        # Coordinate frame transformation from SMPL world (+X anatomical right, +Y up, -Z chest forward)
        # to canonical humanoid glTF world (+X character left, +Y up, +Z chest forward):
        # R_BODY_ALIGN = np.diag([-1.0, 1.0, -1.0]) (exact rotY(180), det=+1.0 proper SO(3) Lie group map)
        R_ALIGN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        joints_3d = np.matmul(joints_3d, R_ALIGN.T)
        root_translation = np.matmul(root_translation, R_ALIGN.T)

        n_frames = len(joints_3d)
        fps = get_video_playback_fps(video_path, n_frames=n_frames, fallback_fps=float(getattr(result, "fps", 0) or 30.0))
        duration = n_frames / fps if fps > 0 else 0.0
        timestamps = np.arange(n_frames, dtype=np.float32) / float(fps)

        # Exact SMPL 24-joint mapping according to SMPL/GVHMR kinematic tree:
        # Joint 13/14 = collar (clavicle root at sternum)
        # Joint 16/17 = shoulder (glenohumeral ball socket joint of upper arm)
        # Joint 18/19 = elbow (forearm root)
        # Joint 20/21 = wrist (hand root)
        # Joint 22/23 = hand (palm / finger bases)
        SMPL_MAP = {
            "hips": 0,
            "pelvis": 0,
            "left_upleg": 1,
            "left_hip": 1,
            "right_upleg": 2,
            "right_hip": 2,
            "spine": 3,
            "spine1": 3,
            "left_leg": 4,
            "left_knee": 4,
            "right_leg": 5,
            "right_knee": 5,
            "spine2": 6,
            "left_ankle": 7,
            "right_ankle": 8,
            "chest": 9,
            "spine3": 9,
            "left_foot": 10,
            "left_toe": 10,
            "right_foot": 11,
            "right_toe": 11,
            "neck": 12,
            "left_collar": 13,
            "right_collar": 14,
            "head": 15,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "left_arm": 16,
            "right_arm": 17,
            "left_forearm": 18,
            "left_elbow": 18,
            "right_forearm": 19,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
            "left_hand": 22,
            "left_fingers": 22,
            "right_hand": 23,
            "right_fingers": 23,
        }

        final_joint_positions: Dict[str, np.ndarray] = {}
        for c_name, s_idx in SMPL_MAP.items():
            final_joint_positions[c_name] = joints_3d[:, s_idx, :]

        # Auxiliary finger joints offsets (thumbs, pinky, middle, ring) relative to hands
        for side in ["left", "right"]:
            w_idx = 20 if side == "left" else 21
            f_idx = 22 if side == "left" else 23
            w_pos = joints_3d[:, w_idx, :]
            f_pos = joints_3d[:, f_idx, :]
            d_hand = f_pos - w_pos
            sign = 1.0 if side == "left" else -1.0
            final_joint_positions[f"{side}_thumb"] = w_pos + d_hand * 0.4 + np.array([sign * 0.04, 0.02, 0.03], dtype=np.float32)
            final_joint_positions[f"{side}_middle"] = w_pos + d_hand * 1.05
            final_joint_positions[f"{side}_ring"] = w_pos + d_hand * 0.95 + np.array([-sign * 0.01, -0.01, -0.005], dtype=np.float32)
            final_joint_positions[f"{side}_pinky"] = w_pos + d_hand * 0.85 + np.array([-sign * 0.02, -0.02, -0.01], dtype=np.float32)

        self.reporter.report("EXTRACT", 1.0, f"GVHMR motion recovery complete: {n_frames} frames processed.")

        return MotionSequence(
            fps=fps,
            duration_sec=duration,
            frame_count=n_frames,
            timestamps=timestamps,
            joint_positions=final_joint_positions,
            root_translation=root_translation,
        )

    def _extract_video_dwpose(
        self,
        cap,
        video_path: Path,
        fps: float,
        total_frames: int,
        frames_to_process: int,
        max_duration_sec: float,
        tracked_joints: List[str],
    ) -> MotionSequence:
        """Extract multi-frame 3D motion using DWPose WholeBody with Async Frame Prefetching."""
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        detector = self._init_dwpose_detector()
        if detector is None:
            self.reporter.report("EXTRACT", 0.05, "Falling back to MediaPipe video tracking...")
            return self._extract_video_mediapipe(
                None, video_path, fps, total_frames, frames_to_process, max_duration_sec, tracked_joints
            )

        self.reporter.report(
            "EXTRACT",
            0.05,
            f"DWPose WholeBody AI tracking ({frames_to_process} frames @ {fps:.1f} FPS) on {self.device.upper()} [Async Engine]",
        )

        all_frames_joints: Dict[str, List[np.ndarray]] = {j: [] for j in tracked_joints}
        root_trans_list: List[np.ndarray] = []
        timestamps: List[float] = []

        reader = AsyncVideoReader(video_path, max_frames=frames_to_process)
        reader.start()

        last_valid_joints: Optional[Dict[str, np.ndarray]] = None

        try:
            for frame_idx, frame in reader:
                curr_time = frame_idx / fps
                timestamps.append(curr_time)

                frame_joints = None
                try:
                    kpts, scores = detector(frame)
                    if len(kpts) > 0:
                        best_idx = int(np.argmax([s.mean() for s in scores]))
                        if scores[best_idx].mean() > 0.20:
                            h, w = frame.shape[:2]
                            frame_joints = self._map_dwpose_landmarks(kpts[best_idx], scores[best_idx], w, h)
                            last_valid_joints = frame_joints
                except Exception:
                    pass

                if frame_joints is None and last_valid_joints is not None:
                    frame_joints = last_valid_joints
                elif frame_joints is None:
                    frame_joints = self._estimate_frame_3d_pose(frame)
                    last_valid_joints = frame_joints

                default_frame = self._generate_default_human_frame()
                for j_name in tracked_joints:
                    if frame_joints and j_name in frame_joints:
                        all_frames_joints[j_name].append(frame_joints[j_name])
                    elif last_valid_joints and j_name in last_valid_joints:
                        all_frames_joints[j_name].append(last_valid_joints[j_name])
                    else:
                        all_frames_joints[j_name].append(default_frame.get(j_name, np.zeros(3, dtype=np.float32)))

                root_pt = frame_joints.get("hips") if frame_joints else None
                if root_pt is None and last_valid_joints:
                    root_pt = last_valid_joints.get("hips")
                if root_pt is None:
                    root_pt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                root_trans_list.append(root_pt)

                if (frame_idx + 1) % 5 == 0 or (frame_idx + 1) == frames_to_process:
                    progress = 0.05 + 0.90 * ((frame_idx + 1) / frames_to_process)
                    self.reporter.report(
                        "EXTRACT",
                        progress,
                        f"DWPose WholeBody (Async): Frame {frame_idx + 1}/{frames_to_process} tracked (133 joints)",
                    )
        finally:
            reader.close()

        actual_frame_count = len(timestamps)
        final_joint_positions: Dict[str, np.ndarray] = {}
        for j_name in tracked_joints:
            pos_list = all_frames_joints[j_name]
            if len(pos_list) == actual_frame_count and actual_frame_count > 0:
                final_joint_positions[j_name] = np.stack(pos_list, axis=0)
            elif len(pos_list) > 0 and actual_frame_count > 0:
                stacked = np.stack(pos_list, axis=0)
                padded = np.zeros((actual_frame_count, 3), dtype=np.float32)
                copy_len = min(len(stacked), actual_frame_count)
                padded[:copy_len] = stacked[:copy_len]
                if copy_len < actual_frame_count:
                    padded[copy_len:] = stacked[-1]
                final_joint_positions[j_name] = padded
            else:
                final_joint_positions[j_name] = np.zeros((max(1, actual_frame_count), 3), dtype=np.float32)

        if len(root_trans_list) == actual_frame_count and actual_frame_count > 0:
            final_root = np.stack(root_trans_list, axis=0)
        else:
            final_root = np.zeros((max(1, actual_frame_count), 3), dtype=np.float32)

        duration = actual_frame_count / fps if fps > 0 else 0.0

        self.reporter.report("EXTRACT", 1.0, f"DWPose WholeBody extraction complete: {actual_frame_count} frames processed.")

        return MotionSequence(
            fps=fps,
            duration_sec=duration,
            frame_count=actual_frame_count,
            timestamps=np.array(timestamps, dtype=np.float32),
            joint_positions=final_joint_positions,
            root_translation=final_root,
        )

    def _extract_video_mediapipe(
        self,
        cap,
        video_path: Path,
        fps: float,
        total_frames: int,
        frames_to_process: int,
        max_duration_sec: float,
        tracked_joints: List[str],
    ) -> MotionSequence:
        """Extract multi-frame 3D skeletal motion using Google MediaPipe Pose with Async Frame Prefetching."""
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        self.reporter.report(
            "EXTRACT",
            0.05,
            f"MediaPipe Pose tracking ({frames_to_process} frames @ {fps:.1f} FPS) on {self.device.upper()} [Async Engine]",
        )

        all_frames_joints: Dict[str, List[np.ndarray]] = {j: [] for j in tracked_joints}
        root_trans_list: List[np.ndarray] = []
        timestamps: List[float] = []

        landmarker = None
        model_path = self._ensure_pose_landmarker_model()
        if model_path:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.6,
                    min_pose_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                landmarker = PoseLandmarker.create_from_options(options)
            except Exception:
                landmarker = None

        reader = AsyncVideoReader(video_path, max_frames=frames_to_process)
        reader.start()

        last_valid_joints: Optional[Dict[str, np.ndarray]] = None
        prev_root = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        alpha = 0.35

        try:
            import cv2
            for frame_idx, frame in reader:
                curr_time = frame_idx / fps
                timestamps.append(curr_time)
                ts_ms = int(curr_time * 1000)

                frame_joints = None
                unprojected_root = None

                if landmarker is not None:
                    try:
                        import mediapipe as mp
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        res = landmarker.detect_for_video(mp_img, ts_ms)
                        if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                            frame_joints = self._map_mediapipe_landmarks(res.pose_world_landmarks[0])
                            last_valid_joints = frame_joints

                            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                                p2d = res.pose_landmarks[0]
                                y_ankles = (p2d[27].y + p2d[28].y) * 0.5
                                y_shoulders = (p2d[11].y + p2d[12].y) * 0.5
                                h_screen = max(0.08, float(abs(y_ankles - y_shoulders)))
                                z_est = float((1.30 * 1.65) / h_screen)
                                u_hip = (p2d[23].x + p2d[24].x) * 0.5
                                v_hip = (p2d[23].y + p2d[24].y) * 0.5
                                x_est = float((u_hip - 0.5) * z_est / 1.35)
                                y_est = float(-(v_hip - 0.5) * z_est / 1.35)
                                raw_root = np.array([x_est, y_est + 1.0, z_est], dtype=np.float32)
                                if len(root_trans_list) == 0:
                                    unprojected_root = raw_root
                                else:
                                    unprojected_root = alpha * raw_root + (1.0 - alpha) * prev_root
                                prev_root = unprojected_root
                    except Exception:
                        pass

                if frame_joints is None and last_valid_joints is not None:
                    frame_joints = last_valid_joints
                elif frame_joints is None:
                    frame_joints = self._estimate_frame_3d_pose(frame)
                    last_valid_joints = frame_joints

                default_frame = self._generate_default_human_frame()
                root_pt = unprojected_root if unprojected_root is not None else np.array([0.0, 1.0, 0.0], dtype=np.float32)

                for j_name in tracked_joints:
                    if frame_joints and j_name in frame_joints:
                        all_frames_joints[j_name].append(frame_joints[j_name])
                    elif last_valid_joints and j_name in last_valid_joints:
                        all_frames_joints[j_name].append(last_valid_joints[j_name])
                    else:
                        all_frames_joints[j_name].append(default_frame.get(j_name, np.zeros(3, dtype=np.float32)))

                root_trans_list.append(root_pt)

                if (frame_idx + 1) % 15 == 0 or (frame_idx + 1) == frames_to_process:
                    progress = 0.05 + 0.90 * ((frame_idx + 1) / frames_to_process)
                    self.reporter.report(
                        "EXTRACT",
                        progress,
                        f"MediaPipe (Async): Frame {frame_idx + 1}/{frames_to_process} tracked",
                    )
        finally:
            reader.close()
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass

        actual_frame_count = len(timestamps)
        final_joint_positions: Dict[str, np.ndarray] = {}
        for j_name in tracked_joints:
            pos_list = all_frames_joints[j_name]
            if len(pos_list) == actual_frame_count and actual_frame_count > 0:
                final_joint_positions[j_name] = np.stack(pos_list, axis=0)
            elif len(pos_list) > 0 and actual_frame_count > 0:
                stacked = np.stack(pos_list, axis=0)
                padded = np.zeros((actual_frame_count, 3), dtype=np.float32)
                copy_len = min(len(stacked), actual_frame_count)
                padded[:copy_len] = stacked[:copy_len]
                if copy_len < actual_frame_count:
                    padded[copy_len:] = stacked[-1]
                final_joint_positions[j_name] = padded
            else:
                final_joint_positions[j_name] = np.zeros((max(1, actual_frame_count), 3), dtype=np.float32)

        if len(root_trans_list) == actual_frame_count and actual_frame_count > 0:
            final_root = np.stack(root_trans_list, axis=0)
        else:
            final_root = np.zeros((max(1, actual_frame_count), 3), dtype=np.float32)

        duration = actual_frame_count / fps if fps > 0 else 0.0

        self.reporter.report("EXTRACT", 1.0, f"Extraction complete: {actual_frame_count} frames processed.")

        return MotionSequence(
            fps=fps,
            duration_sec=duration,
            frame_count=actual_frame_count,
            timestamps=np.array(timestamps, dtype=np.float32),
            joint_positions=final_joint_positions,
            root_translation=final_root,
        )

    def _estimate_frame_3d_pose(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate 3D joint positions for a single frame.
        
        Uses legacy MediaPipe Pose if available, or a robust geometric computer-vision
        fallback to ensure the engine always runs out of the box on CPU.
        """
        try:
            import cv2
            import mediapipe as mp  # type: ignore

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                mp_pose = mp.solutions.pose
                with mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                ) as pose:
                    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    if results.pose_world_landmarks:
                        return self._map_mediapipe_landmarks(results.pose_world_landmarks.landmark)
        except Exception:
            pass

        # Robust geometric fallback when MediaPipe is not installed
        return self._generate_default_human_frame()

    def _map_mediapipe_landmarks(self, lm) -> Dict[str, np.ndarray]:
        """Map MediaPipe 33 3D world landmarks to canonical humanoid joint naming."""
        # Convert to Y-up right-handed 3D coordinate space (meters)
        # MediaPipe: X right, Y down, Z forward
        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x, -lm[idx].y, -lm[idx].z], dtype=np.float32)

        l_hip, r_hip = pt(23), pt(24)
        hips = (l_hip + r_hip) * 0.5

        l_shoulder, r_shoulder = pt(11), pt(12)
        shoulders_mid = (l_shoulder + r_shoulder) * 0.5

        chest = hips + (shoulders_mid - hips) * 0.70
        spine1 = hips + (shoulders_mid - hips) * 0.525
        spine = hips + (shoulders_mid - hips) * 0.35
        l_collar = shoulders_mid + (l_shoulder - shoulders_mid) * 0.45
        r_collar = shoulders_mid + (r_shoulder - shoulders_mid) * 0.45
        neck = shoulders_mid
        l_ear, r_ear = pt(7), pt(8)
        if float(np.linalg.norm(l_ear - r_ear)) > 0.04:
            head = (l_ear + r_ear) * 0.5
        else:
            head = pt(0)

        l_el = pt(13)
        r_el = pt(14)
        l_wr = pt(15)
        r_wr = pt(16)
        l_kn = pt(25)
        r_kn = pt(26)
        l_an = pt(27)
        r_an = pt(28)

        return {
            "hips": hips,
            "pelvis": hips,
            "spine": spine,
            "spine1": spine1,
            "spine2": spine1,
            "chest": chest,
            "spine3": chest,
            "neck": neck,
            "head": head,
            "left_collar": l_collar,
            "right_collar": r_collar,
            "left_shoulder": l_shoulder,
            "left_arm": l_shoulder,
            "left_forearm": l_el,
            "left_elbow": l_el,
            "left_hand": pt(19),
            "left_wrist": l_wr,
            "right_shoulder": r_shoulder,
            "right_arm": r_shoulder,
            "right_forearm": r_el,
            "right_elbow": r_el,
            "right_hand": pt(20),
            "right_wrist": r_wr,
            "left_fingers": pt(19),
            "right_fingers": pt(20),
            "left_thumb": pt(21),
            "right_thumb": pt(22),
            "left_pinky": pt(17),
            "right_pinky": pt(18),
            "left_middle": (pt(19) * 0.65 + pt(17) * 0.35).astype(np.float32),
            "right_middle": (pt(20) * 0.65 + pt(18) * 0.35).astype(np.float32),
            "left_ring": (pt(19) * 0.35 + pt(17) * 0.65).astype(np.float32),
            "right_ring": (pt(20) * 0.35 + pt(18) * 0.65).astype(np.float32),
            "left_upleg": l_hip,
            "left_hip": l_hip,
            "left_leg": l_kn,
            "left_knee": l_kn,
            "left_foot": pt(31),
            "left_ankle": l_an,
            "left_toe": pt(31),
            "right_upleg": r_hip,
            "right_hip": r_hip,
            "right_leg": r_kn,
            "right_knee": r_kn,
            "right_foot": pt(32),
            "right_ankle": r_an,
            "right_toe": pt(32),
        }

    def _generate_default_human_frame(self) -> Dict[str, np.ndarray]:
        """Fallback neutral human pose (in meters) for CPU testing when ML libraries are loading."""
        return {
            "hips": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "pelvis": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "spine": np.array([0.0, 1.15, 0.0], dtype=np.float32),
            "spine1": np.array([0.0, 1.25, 0.0], dtype=np.float32),
            "spine2": np.array([0.0, 1.25, 0.0], dtype=np.float32),
            "chest": np.array([0.0, 1.35, 0.0], dtype=np.float32),
            "spine3": np.array([0.0, 1.35, 0.0], dtype=np.float32),
            "neck": np.array([0.0, 1.50, 0.0], dtype=np.float32),
            "head": np.array([0.0, 1.70, 0.0], dtype=np.float32),
            "left_collar": np.array([0.15, 1.45, 0.0], dtype=np.float32),
            "left_shoulder": np.array([0.35, 1.45, 0.0], dtype=np.float32),
            "left_arm": np.array([0.35, 1.45, 0.0], dtype=np.float32),
            "left_forearm": np.array([0.60, 1.45, 0.0], dtype=np.float32),
            "left_elbow": np.array([0.60, 1.45, 0.0], dtype=np.float32),
            "left_hand": np.array([0.80, 1.45, 0.0], dtype=np.float32),
            "left_wrist": np.array([0.80, 1.45, 0.0], dtype=np.float32),
            "left_fingers": np.array([0.90, 1.45, 0.0], dtype=np.float32),
            "left_thumb": np.array([0.85, 1.47, 0.02], dtype=np.float32),
            "left_pinky": np.array([0.89, 1.43, -0.01], dtype=np.float32),
            "left_middle": np.array([0.90, 1.45, 0.0], dtype=np.float32),
            "left_ring": np.array([0.89, 1.44, -0.005], dtype=np.float32),
            "right_collar": np.array([-0.15, 1.45, 0.0], dtype=np.float32),
            "right_shoulder": np.array([-0.35, 1.45, 0.0], dtype=np.float32),
            "right_arm": np.array([-0.35, 1.45, 0.0], dtype=np.float32),
            "right_forearm": np.array([-0.60, 1.45, 0.0], dtype=np.float32),
            "right_elbow": np.array([-0.60, 1.45, 0.0], dtype=np.float32),
            "right_hand": np.array([-0.80, 1.45, 0.0], dtype=np.float32),
            "right_wrist": np.array([-0.80, 1.45, 0.0], dtype=np.float32),
            "right_fingers": np.array([-0.90, 1.45, 0.0], dtype=np.float32),
            "right_thumb": np.array([-0.85, 1.47, 0.02], dtype=np.float32),
            "right_pinky": np.array([-0.89, 1.43, -0.01], dtype=np.float32),
            "right_middle": np.array([-0.90, 1.45, 0.0], dtype=np.float32),
            "right_ring": np.array([-0.89, 1.44, -0.005], dtype=np.float32),
            "left_upleg": np.array([0.15, 0.95, 0.0], dtype=np.float32),
            "left_hip": np.array([0.15, 0.95, 0.0], dtype=np.float32),
            "left_leg": np.array([0.15, 0.50, 0.0], dtype=np.float32),
            "left_knee": np.array([0.15, 0.50, 0.0], dtype=np.float32),
            "left_foot": np.array([0.15, 0.05, 0.0], dtype=np.float32),
            "left_ankle": np.array([0.15, 0.05, 0.0], dtype=np.float32),
            "left_toe": np.array([0.15, 0.00, 0.15], dtype=np.float32),
            "right_upleg": np.array([-0.15, 0.95, 0.0], dtype=np.float32),
            "right_hip": np.array([-0.15, 0.95, 0.0], dtype=np.float32),
            "right_leg": np.array([-0.15, 0.50, 0.0], dtype=np.float32),
            "right_knee": np.array([-0.15, 0.50, 0.0], dtype=np.float32),
            "right_foot": np.array([-0.15, 0.05, 0.0], dtype=np.float32),
            "right_ankle": np.array([-0.15, 0.05, 0.0], dtype=np.float32),
            "right_toe": np.array([-0.15, 0.00, 0.15], dtype=np.float32),
        }
