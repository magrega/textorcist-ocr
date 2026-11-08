"""Load the game's sprite-font glyphs used to read the battle parchment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
GLYPHS_ROOT = ROOT / "glyphs" / "sprite_fonts"

# The battle parchment is drawn with the gothic Alagard sprite font.
DEFAULT_PACKS = ("fontAlagard",)


@dataclass
class GlyphPack:
    name: str
    templates: dict[str, np.ndarray]  # binary 0/255, native size
    space_width: int


def _load_pack(name: str) -> GlyphPack | None:
    folder = GLYPHS_ROOT / name
    meta_path = folder / "metrics.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    templates: dict[str, np.ndarray] = {}
    widths: list[int] = []
    for g in meta.get("glyphs", []):
        ch = g.get("char") or ""
        if not isinstance(ch, str) or len(ch) != 1:
            continue
        if not ch.isascii():
            continue
        path = folder / g["file"]
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            mask = (raw[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
            mask = (gray > 32).astype(np.uint8) * 255
        if int(mask.sum()) < 255:
            continue
        templates[ch] = mask
        if ch.isalnum():
            widths.append(mask.shape[1])
    if len(templates) < 20:
        return None
    space_w = int(np.median(widths)) if widths else 8
    return GlyphPack(name=name, templates=templates, space_width=space_w)


@lru_cache(maxsize=1)
def load_primary_pack() -> GlyphPack | None:
    preferred: list[str] = []
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            name = cfg.get("glyph_pack")
            if isinstance(name, str) and name.strip():
                preferred.append(name.strip())
        except Exception:  # noqa: BLE001
            pass
    for name in [*preferred, *DEFAULT_PACKS]:
        pack = _load_pack(name)
        if pack is not None:
            return pack
    return None
