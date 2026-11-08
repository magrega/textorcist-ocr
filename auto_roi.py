"""Locate the bottom-center battle parchment from its pixels, not its resolution."""
from __future__ import annotations

import cv2
import numpy as np


def parchment_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (channel.astype(np.int16) for channel in cv2.split(bgr))
    return (
        (r >= 190) & (g >= 170) & (b >= 120)
        & (r - b >= 15) & (r - b <= 100) & (r - g <= 45)
    ).astype(np.uint8)


def find_parchment(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x, y, width, height in physical game-window pixels.

    Work at a bounded resolution so 720p and 4K have the same geometry and
    search cost. A broad parchment interior is required; text alone or a small
    beige object cannot activate automatic capture.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    if width < 320 or height < 200:
        return None
    scale = min(1.0, 1600.0 / width)
    sample = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = sample.shape[:2]
    mask = parchment_mask(sample)
    start = round(sh * 0.72)
    mask[:start] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for x, y, w, h, _area in stats[1:count]:
        if w < sw * 0.40 or h < sh * 0.025 or w / h < 5:
            continue
        interior = mask[y:y + h, x:x + w]
        if float(interior.mean()) < 0.45:
            continue
        center = (x + w / 2) / sw
        # The panel may extend behind the bottom edge of the game window.
        candidates.append((abs(center - 0.5) * 3 + abs(y + h - sh) / sh * 0.1, x, y, w, h))
    if not candidates:
        return None
    _, x, y, w, h = min(candidates)
    pad_x, pad_y = max(2, round(sw * 0.002)), max(2, round(sh * 0.004))
    return (
        round(max(0, x - pad_x) / scale),
        round(max(0, y - pad_y) / scale),
        round(min(sw, x + w + pad_x) / scale) - round(max(0, x - pad_x) / scale),
        round(min(sh, y + h + pad_y) / scale) - round(max(0, y - pad_y) / scale),
    )
