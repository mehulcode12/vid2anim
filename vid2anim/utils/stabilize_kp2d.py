"""Keep COCO-17 left/right identity consistent across time.

RTMPose/ViTPose label left/right independently per frame. When the subject
turns to face away, those labels often swap. GVHMR then reads a jumping 2D
skeleton and spins the 3D heading. This assigns each left/right pair to the
previous frame's positions so anatomical identity is tracked, not re-guessed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np

# Full COCO-17 left/right permutation (nose stays at 0).
COCO17_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], dtype=np.int64)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _lr_x(frame: np.ndarray) -> float:
    """Shoulder left-minus-right in image x. Hips if shoulders are collapsed."""
    sh = float(frame[5, 0] - frame[6, 0])
    if abs(sh) >= 40.0:
        return sh
    return float(frame[11, 0] - frame[12, 0])


def stabilize_coco17(kp2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (stabilized (F,17,3), per-frame swap flags (F,)).

    A real 180-degree turn drives left-right x through a narrow side view. A
    label flip jumps the sign while the width stays large. Those jumps are
    un-swapped so GVHMR sees one anatomical identity.
    """
    src = np.asarray(_as_numpy(kp2d), dtype=np.float64)
    if src.ndim != 3 or src.shape[1] < 17:
        raise ValueError(f"expected (F, 17, 3) COCO-17, got {src.shape}")
    out = src.copy()
    n = len(out)
    narrow = float(np.clip(0.45 * np.median([abs(_lr_x(f)) for f in src]), 40.0, 80.0))
    swapped = np.zeros(n, dtype=np.int32)
    last_sign = np.sign(_lr_x(out[0])) or 1.0
    last_turn = -(10**9)
    for i in range(n):
        raw = src[i]
        width = abs(_lr_x(raw))
        sign = np.sign(_lr_x(raw)) or last_sign
        if width >= 40.0 and sign != last_sign:
            start = max(0, last_turn + 1, i - 18)
            widths = [abs(_lr_x(f)) for f in out[start:i]]
            run = best = 0
            for w in widths:
                run = run + 1 if w < narrow else 0
                best = max(best, run)
            if best >= (1 if last_turn < 0 else 4):
                last_sign = sign
                last_turn = i
            else:
                raw = raw[COCO17_SWAP]
                swapped[i] = 1
        out[i] = raw
        sh = out[i, 5, 0] - out[i, 6, 0]
        hip = out[i, 11, 0] - out[i, 12, 0]
        if abs(sh) >= 40.0 and abs(hip) >= 40.0 and np.sign(sh) != np.sign(hip):
            out[i, [11, 12, 13, 14, 15, 16]] = out[i, [12, 11, 14, 13, 16, 15]]
    return out.astype(np.float32), swapped


def stabilize_pose_cache(pose_cache: Path) -> bool:
    """Stabilize GVHMR's 2D-pose cache in place. True if the file changed.

    Idempotent: a cache that was already stabilized comes back unchanged. The
    pre-stabilization data is kept next to it (*_raw) so estimator-vs-
    stabilizer questions stay answerable after the fact.
    """
    if not pose_cache.exists():
        return False
    import torch

    try:
        current = torch.load(pose_cache, weights_only=False)
        fixed, swaps = stabilize_coco17(current)
        curr_np = _as_numpy(current).astype(np.float32)
        if np.allclose(fixed, curr_np):
            return False
        raw_backup = pose_cache.with_name(pose_cache.stem + "_raw" + pose_cache.suffix)
        if not raw_backup.exists():
            shutil.copy2(pose_cache, raw_backup)
        torch.save(torch.from_numpy(fixed), pose_cache)
        return True
    except Exception:
        return False
