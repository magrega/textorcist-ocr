"""Battle-text helpers: parchment gate + glyph-rendered phrase verification."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
GLYPHS_ROOT = ROOT / "glyphs" / "sprite_fonts"

# Combat parchment uses gothic Alagard-style glyphs most strongly in probes.
DEFAULT_PACKS = ("fontAlagard", "fontCeltic2x", "fontAlagardOutline", "fontCeltic")


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


def is_battle_parchment(bgr: np.ndarray) -> bool:
    """
    True only for the beige combat typing strip.
    Rejects dialogue / cutscenes drawn over near-black backgrounds.
    """
    if bgr is None or bgr.size == 0:
        return False
    h, w = bgr.shape[:2]
    if h < 8 or w < 40:
        return False

    mean = bgr.mean(axis=(0, 1))  # B,G,R
    # Dialogue boxes: predominantly black / very dark
    if float(mean.mean()) < 70:
        return False
    # Parchment is warm beige (R high, G mid-high, B lower than R)
    if mean[2] < 140 or mean[1] < 120:
        return False
    if mean[2] + 5 < mean[0]:  # blue-ish overall → not parchment
        return False

    # Fraction of beige paper pixels vs near-black
    B, G, R = cv2.split(bgr)
    beige = (
        (R > 150)
        & (G > 130)
        & (R >= G - 10)
        & (G >= B - 5)
        & ((R.astype(np.int16) - B.astype(np.int16)) > 20)
    )
    dark = (R.astype(np.int16) + G.astype(np.int16) + B.astype(np.int16)) < 90
    beige_frac = float(beige.mean())
    dark_frac = float(dark.mean())
    if dark_frac > 0.55 and beige_frac < 0.25:
        return False
    if beige_frac < 0.20:
        return False
    return True


def clean_ink_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    out = (mask > 0).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 20), 1))
    horiz = cv2.morphologyEx(out, cv2.MORPH_OPEN, hk)
    out = cv2.subtract(out, horiz)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(14, h // 2)))
    vert = cv2.morphologyEx(out, cv2.MORPH_OPEN, vk)
    out = cv2.subtract(out, vert)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return out


def ink_content_band(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop to the densest horizontal ink band. Returns (band, (x1,y1,x2,y2))."""
    mask = clean_ink_mask(mask)
    if mask is None or int((mask > 0).sum()) < 60:
        return None
    row = (mask > 0).sum(axis=1)
    if row.max() < 4:
        return None
    peak = int(row.argmax())
    # grow while density stays meaningful
    thr = max(3, int(row.max() * 0.12))
    y1 = peak
    while y1 > 0 and row[y1 - 1] >= thr:
        y1 -= 1
    y2 = peak
    while y2 < len(row) - 1 and row[y2 + 1] >= thr:
        y2 += 1
    y2 += 1
    # clamp absurdly tall bands (decorations)
    if y2 - y1 > 120:
        y1, y2 = max(0, peak - 50), min(mask.shape[0], peak + 50)
    band = mask[y1:y2]
    col = (band > 0).sum(axis=0)
    xs = np.where(col > 0)[0]
    if len(xs) < 4:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return band[:, x1:x2], (x1, y1, x2, y2)


def render_text_mask(pack: GlyphPack, text: str, scale: float) -> np.ndarray | None:
    """Render phrase as a binary mask using sprite-font glyphs."""
    if not text or scale <= 0:
        return None
    space = max(2, int(round(pack.space_width * scale)))
    letter_gap = max(1, int(round(1 * scale)))
    pieces: list[tuple[str, np.ndarray | None, int]] = []  # kind, glyph, advance
    max_h = 0
    for ch in text:
        if ch == " ":
            pieces.append(("space", None, space))
            continue
        tmpl = pack.templates.get(ch)
        if tmpl is None:
            tmpl = pack.templates.get(ch.swapcase())
        if tmpl is None:
            tmpl = pack.templates.get(ch.lower())
        if tmpl is None:
            tmpl = pack.templates.get(ch.upper())
        if tmpl is None:
            pieces.append(("space", None, max(2, int(3 * scale))))
            continue
        tw = max(2, int(round(tmpl.shape[1] * scale)))
        th = max(2, int(round(tmpl.shape[0] * scale)))
        rt = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_NEAREST)
        pieces.append(("glyph", rt, tw + letter_gap))
        max_h = max(max_h, th)
    if max_h < 4:
        return None
    total_w = sum(adv for _k, _g, adv in pieces) + 2
    if total_w < 4:
        return None
    canvas = np.zeros((max_h, total_w), np.uint8)
    x = 1
    for kind, glyph, adv in pieces:
        if kind == "glyph" and glyph is not None:
            th, tw = glyph.shape
            y = max(0, (max_h - th) // 2)
            canvas[y : y + th, x : x + tw] = np.maximum(
                canvas[y : y + th, x : x + tw], glyph
            )
        x += adv
    ys, xs = np.where(canvas > 0)
    if len(xs) < 4:
        return None
    return canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return -1.0
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-6
    return float(af.dot(bf) / denom)


def correlate_masks(haystack: np.ndarray, needle: np.ndarray) -> float:
    """Return best normalized correlation of needle inside haystack (or padded)."""
    if haystack is None or needle is None:
        return 0.0
    H, W = haystack.shape[:2]
    h, w = needle.shape[:2]
    if h < 3 or w < 3 or H < 3 or W < 3:
        return 0.0

    # Resize needle to haystack height for scale-robust compare when close
    if abs(h - H) > 1:
        scale = H / h
        nw = max(3, int(round(w * scale)))
        nh = H
        needle = cv2.resize(needle, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = needle.shape

    if w > W:
        # shrink needle to hay width
        needle = cv2.resize(needle, (W, h), interpolation=cv2.INTER_NEAREST)
        h, w = needle.shape
        if w > W or h > H:
            return 0.0

    if h > H:
        needle = cv2.resize(needle, (w, H), interpolation=cv2.INTER_NEAREST)
        h, w = needle.shape

    if w >= W and h >= H:
        return max(0.0, _ncc(haystack, cv2.resize(needle, (W, H), interpolation=cv2.INTER_NEAREST)))

    res = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else 0.0


def estimate_scale(pack: GlyphPack, ink_band: np.ndarray) -> float:
    """Pick a scale where common letters peak against the ink band."""
    probes = [ch for ch in "eEtTaAoOsShHiInNrRuUmM" if ch in pack.templates]
    if not probes:
        probes = list(pack.templates.keys())[:12]
    best_scale, best_score = 4.0, -1.0
    max_scale = max(2.0, (ink_band.shape[0] - 1) / 8.0)
    for scale in np.linspace(2.0, min(8.0, max_scale), 19):
        scores = []
        for ch in probes[:10]:
            tmpl = pack.templates[ch]
            tw = max(2, int(round(tmpl.shape[1] * scale)))
            th = max(2, int(round(tmpl.shape[0] * scale)))
            if th >= ink_band.shape[0] or tw >= ink_band.shape[1]:
                continue
            rt = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(ink_band, rt, cv2.TM_CCOEFF_NORMED)
            scores.append(float(res.max()))
        if not scores:
            continue
        score = float(np.mean(sorted(scores, reverse=True)[:3]))
        if score > best_score:
            best_score, best_scale = score, float(scale)
    return best_scale


def score_phrase_against_ink(
    pack: GlyphPack,
    phrase: str,
    ink_band: np.ndarray,
    scale: float | None = None,
) -> float:
    if not phrase or ink_band is None:
        return 0.0
    scale = scale or estimate_scale(pack, ink_band)
    rendered = render_text_mask(pack, phrase, scale)
    if rendered is None:
        return 0.0
    return max(0.0, correlate_masks(ink_band, rendered)) * 100.0


def best_phrase_by_glyphs(
    ink_mask: np.ndarray,
    candidates: list[str],
    pack: GlyphPack | None = None,
    top_n: int = 40,
) -> tuple[str, float, float]:
    """
    Rank phrase candidates by glyph-render correlation with ink.
    Returns (best_phrase, score_0_100, scale).
    """
    pack = pack or load_primary_pack()
    if pack is None or not candidates:
        return "", 0.0, 0.0
    band_info = ink_content_band(ink_mask)
    if band_info is None:
        return "", 0.0, 0.0
    band, _box = band_info
    scale = estimate_scale(pack, band)

    # Prefer candidates whose rendered width is in the ballpark of the band
    scored: list[tuple[float, str]] = []
    # Limit work: unique, longer-first sample
    uniq = []
    seen = set()
    for p in sorted(candidates, key=lambda s: len(s), reverse=True):
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
        if len(uniq) >= top_n:
            break

    for phrase in uniq:
        sc = score_phrase_against_ink(pack, phrase, band, scale=scale)
        scored.append((sc, phrase))
    if not scored:
        return "", 0.0, scale
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_phrase = scored[0]
    return best_phrase, best_score, scale


def typed_fraction_from_masks(
    typed_mask: np.ndarray,
    remain_mask: np.ndarray,
) -> float:
    t = int((typed_mask > 0).sum())
    r = int((remain_mask > 0).sum())
    total = t + r
    if total < 40:
        return 0.0
    return t / total


def split_phrase_by_fraction(phrase: str, typed_frac: float) -> tuple[str, str]:
    """Split phrase into already/remaining using compact-letter fraction."""
    if not phrase:
        return "", ""
    letters = [(i, ch) for i, ch in enumerate(phrase) if ch.isalnum()]
    if not letters:
        return "", phrase
    cut_n = int(round(typed_frac * len(letters)))
    cut_n = max(0, min(len(letters), cut_n))
    if cut_n <= 0:
        return "", phrase
    if cut_n >= len(letters):
        return phrase, ""
    cut_idx = letters[cut_n][0]
    already = phrase[:cut_idx]
    remain = phrase[cut_idx:]
    # keep leading space on remain tidy
    return already, remain.lstrip() if already.endswith(" ") or remain[:1] == " " else remain
