"""Battle progress from screen ink + glyph word layout.

Typed = white outline around letters (definitive).
Remaining = orange/brown fill without outline.

We align the known phrase with glyphs, score each word's x-span by
white vs orange pixels, and resume at the first brown word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from glyph_matcher import GlyphPack, load_primary_pack, render_text_mask


@dataclass
class ProgressSplit:
    already: str
    remaining: str
    typed_frac: float
    method: str


def parchment_row_gate(bgr: np.ndarray) -> np.ndarray:
    B, G, R = cv2.split(bgr)
    parch = (
        (R > 150)
        & (G > 140)
        & (B > 110)
        & (R >= G - 20)
        & (G >= B - 25)
        & ((R.astype(np.int16) - B.astype(np.int16)) > 8)
    )
    row_frac = parch.mean(axis=1)
    keep = row_frac > 0.35
    if not np.any(keep):
        keep = row_frac > 0.15
    gate = np.zeros(bgr.shape[:2], np.uint8)
    gate[keep] = 255
    gate = cv2.bitwise_and(gate, parch.astype(np.uint8) * 255)
    return cv2.dilate(gate, np.ones((3, 3), np.uint8), iterations=1)


def ink_layers(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (white_outline, orange_remain, dark_fill) uint8 masks.

    White outline must sit next to letter ink — bare parchment cream is NOT typed.
    """
    gate = parchment_row_gate(bgr)
    B, G, R = cv2.split(bgr)
    R16, G16, B16 = R.astype(np.int16), G.astype(np.int16), B.astype(np.int16)
    orange = (
        (R16 > 140)
        & (R16 > G16 + 40)
        & (R16 > B16 + 40)
        & (G16 < 155)
        & (B16 < 155)
    ).astype(np.uint8) * 255
    dark = ((R16 < 75) & (G16 < 75) & (B16 < 110)).astype(np.uint8) * 255
    # Bright pixels alone match parchment; require adjacency to orange/dark ink
    bright = ((R > 220) & (G > 220) & (B > 220)).astype(np.uint8) * 255
    ink_near = cv2.dilate(
        cv2.bitwise_or(orange, dark), np.ones((5, 5), np.uint8), iterations=1
    )
    white = cv2.bitwise_and(bright, ink_near)
    white = cv2.bitwise_and(white, gate)
    orange = cv2.bitwise_and(orange, gate)
    dark = cv2.bitwise_and(dark, gate)
    # Drop long horizontal guide-lines from orange
    h, w = orange.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(28, w // 16), 1))
    orange = cv2.subtract(orange, cv2.morphologyEx(orange, cv2.MORPH_OPEN, hk))
    return white, orange, dark


def split_battle_ink(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility: typed ≈ outline|dark-near-outline, remain ≈ orange."""
    white, orange, dark = ink_layers(bgr)
    near_white = cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=1)
    typed = cv2.bitwise_or(white, cv2.bitwise_and(dark, near_white))
    remain = cv2.bitwise_and(orange, cv2.bitwise_not(near_white))
    return typed, remain, cv2.bitwise_or(typed, orange)


def _ink_band_box(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    if mask is None or int((mask > 0).sum()) < 20:
        return None
    row = (mask > 0).sum(axis=1)
    if row.max() < 2:
        return None
    peak = int(row.argmax())
    thr = max(2, int(row.max() * 0.08))
    y1 = peak
    while y1 > 0 and row[y1 - 1] >= thr:
        y1 -= 1
    y2 = peak
    while y2 < len(row) - 1 and row[y2 + 1] >= thr:
        y2 += 1
    y2 += 1
    y1 = max(0, y1 - 4)
    y2 = min(mask.shape[0], y2 + 4)
    col = (mask[y1:y2] > 0).sum(axis=0)
    xs = np.where(col > 0)[0]
    if len(xs) < 3:
        return None
    return int(xs.min()), y1, int(xs.max()) + 1, y2


def _word_spans(phrase: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", phrase)]


def _render_char_x_positions(
    pack: GlyphPack, phrase: str, scale: float
) -> list[tuple[str, int, int]] | None:
    space = max(2, int(round(pack.space_width * scale)))
    letter_gap = max(1, int(round(1 * scale)))
    x = 0
    spans: list[tuple[str, int, int]] = []
    max_h = 0
    glyphs: list[tuple[str, np.ndarray | None, int]] = []
    for ch in phrase:
        if ch == " ":
            glyphs.append((" ", None, space))
            continue
        tmpl = pack.templates.get(ch)
        if tmpl is None:
            tmpl = pack.templates.get(ch.swapcase())
        if tmpl is None:
            tmpl = pack.templates.get(ch.lower())
        if tmpl is None:
            tmpl = pack.templates.get(ch.upper())
        if tmpl is None:
            glyphs.append((ch, None, max(2, int(3 * scale))))
            continue
        tw = max(2, int(round(tmpl.shape[1] * scale)))
        th = max(2, int(round(tmpl.shape[0] * scale)))
        rt = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_NEAREST)
        glyphs.append((ch, rt, tw + letter_gap))
        max_h = max(max_h, th)
    if max_h < 4:
        return None
    for ch, glyph, adv in glyphs:
        if ch == " ":
            spans.append((ch, x, x + adv))
            x += adv
            continue
        w = glyph.shape[1] if glyph is not None else max(2, adv)
        spans.append((ch, x, x + w))
        x += adv
    return spans


def _scale_to_width(pack: GlyphPack, phrase: str, target_w: int) -> float:
    lo, hi = 1.0, 8.0
    best = 2.0
    for _ in range(10):  # fewer iters = faster
        mid = (lo + hi) / 2
        rend = render_text_mask(pack, phrase, mid)
        if rend is None:
            lo = mid
            continue
        if rend.shape[1] < target_w:
            lo = mid
        else:
            hi = mid
        best = mid
    return float(best)


def _column_cut_x(white_b: np.ndarray, orange_b: np.ndarray) -> int | None:
    """First x where orange dominates after white-outline typed region."""
    wcol = (white_b > 0).sum(axis=0).astype(np.float32)
    ocol = (orange_b > 0).sum(axis=0).astype(np.float32)
    if wcol.size < 8:
        return None
    k = np.ones(9, np.float32) / 9.0
    ts = np.convolve(wcol, k, mode="same")
    rs = np.convolve(ocol, k, mode="same")
    seen_typed = False
    for x in range(len(ts)):
        if ts[x] >= 5:
            seen_typed = True
        if not seen_typed:
            continue
        if rs[x] >= max(5.0, ts[x] * 1.15):
            end = min(len(ts), x + 18)
            if rs[x:end].mean() > ts[x:end].mean() + 0.5:
                return x
    return None


def _split_at_letter_frac(phrase: str, frac: float) -> tuple[str, str]:
    """Cut phrase at letter-fraction, snapping forward to a word start."""
    letters = [(i, ch) for i, ch in enumerate(phrase) if ch.isalnum()]
    if not letters:
        return "", phrase
    frac = max(0.0, min(1.0, frac))
    cut_n = int(round(frac * len(letters)))
    cut_n = max(0, min(len(letters), cut_n))
    if cut_n <= 0:
        return "", phrase
    if cut_n >= len(letters):
        return phrase, ""
    # Index in phrase of the cut_n-th letter
    idx = letters[cut_n][0]
    # If mid-word, snap forward to next word (don't retype a half-typed word)
    if idx > 0 and phrase[idx - 1] != " ":
        j = idx
        while j < len(phrase) and phrase[j] != " ":
            j += 1
        while j < len(phrase) and phrase[j] == " ":
            j += 1
        idx = j
    already = phrase[:idx].rstrip()
    remaining = phrase[idx:].lstrip()
    return already, remaining


def split_progress_by_glyphs(
    bgr: np.ndarray,
    phrase: str,
    pack: GlyphPack | None = None,
) -> ProgressSplit:
    """Resume using white-outline → orange column cut (robust on long lines)."""
    phrase = re.sub(r"\s+", " ", (phrase or "").strip())
    phrase = re.sub(r"\.{2,}$", "", phrase).strip()
    if not phrase:
        return ProgressSplit("", "", 0.0, "empty")

    pack = pack or load_primary_pack()
    white, orange, dark = ink_layers(bgr)
    white_px = int((white > 0).sum())
    orange_px = int((orange > 0).sum())
    typed_frac = (white_px / max(1, white_px + orange_px)) if (white_px + orange_px) else 0.0

    if orange_px < 35 and white_px > 60:
        return ProgressSplit(phrase, "", typed_frac, "all_typed")
    # Fresh line: orange dominates — never invent a typed prefix from parchment noise
    if typed_frac < 0.18 and orange_px > 60:
        return ProgressSplit("", phrase, typed_frac, "all_remain")
    if white_px < 40 and orange_px > 80:
        return ProgressSplit("", phrase, typed_frac, "all_remain")

    box = _ink_band_box(orange) if orange_px > 40 else None
    if box is None:
        box = _ink_band_box(cv2.bitwise_or(white, cv2.bitwise_or(orange, dark)))
    if box is None:
        # No reliable span — prefer full remain over guessing a cut
        return ProgressSplit("", phrase, typed_frac, "no_span")

    _bx1, by1, _bx2, by2 = box
    pad_y = max(8, (by2 - by1) // 2)
    by1 = max(0, by1 - pad_y)
    by2 = min(bgr.shape[0], by2 + pad_y)
    combo = cv2.bitwise_or(white, cv2.bitwise_or(orange, dark))[by1:by2]
    xs = np.where(combo.sum(axis=0) > 0)[0]
    if len(xs) < 3:
        return ProgressSplit("", phrase, typed_frac, "no_span")
    w = bgr.shape[1]
    x_lo = max(int(xs.min()), int(w * 0.06))
    x_hi = min(int(xs.max()) + 1, int(w * 0.94))
    if x_hi <= x_lo + 10:
        x_lo, x_hi = int(xs.min()), int(xs.max()) + 1

    white_b = white[by1:by2, x_lo:x_hi]
    orange_b = orange[by1:by2, x_lo:x_hi]
    band_w = x_hi - x_lo

    # Primary: column transition (works on long mercy/hope lines)
    cut_x = _column_cut_x(white_b, orange_b)
    if cut_x is not None:
        # Tiny left cut on a short phrase is almost always chrome noise
        frac = cut_x / max(1, band_w)
        if frac < 0.12 and typed_frac < 0.25:
            return ProgressSplit("", phrase, typed_frac, "all_remain")
        already, remaining = _split_at_letter_frac(phrase, frac)
        # Single short word "already" on a still-orange line → ignore
        if already and remaining and typed_frac < 0.22 and len(already.split()) <= 1:
            return ProgressSplit("", phrase, typed_frac, "all_remain")
        return ProgressSplit(already, remaining, typed_frac, "column_cut")

    # Fallback: per-word white vs orange via glyph layout
    if pack is None:
        return ProgressSplit("", phrase, typed_frac, "no_pack")
    scale = _scale_to_width(pack, phrase, band_w)
    char_spans = _render_char_x_positions(pack, phrase, scale)
    if not char_spans:
        return ProgressSplit("", phrase, typed_frac, "render_fail")
    rend_w = char_spans[-1][2]
    if rend_w < 4:
        return ProgressSplit("", phrase, typed_frac, "render_fail")
    sx = band_w / rend_w

    words = _word_spans(phrase)
    resume_idx = len(words)
    for wi, (_word, a, b) in enumerate(words):
        x0 = max(0, int(char_spans[a][1] * sx) - 1)
        x1 = min(band_w, int(char_spans[b - 1][2] * sx) + 1)
        if x1 <= x0 + 1:
            continue
        w_n = int((white_b[:, x0:x1] > 0).sum())
        o_n = int((orange_b[:, x0:x1] > 0).sum())
        if w_n >= 40:
            continue
        if o_n >= 25 and w_n < 40:
            resume_idx = wi
            break
        if o_n >= 15 and o_n > w_n * 2:
            resume_idx = wi
            break

    if resume_idx <= 0:
        already, remaining = "", phrase
    elif resume_idx >= len(words):
        already, remaining = phrase, ""
    else:
        cut_at = words[resume_idx][1]
        already = phrase[:cut_at].rstrip()
        remaining = phrase[cut_at:].lstrip()

    return ProgressSplit(already, remaining, typed_frac, "glyph_layout")


def progress_only(bgr: np.ndarray, phrase: str) -> ProgressSplit:
    """Fast path: known phrase → progress split only (no OCR)."""
    return split_progress_by_glyphs(bgr, phrase)


def ink_fingerprint(bgr: np.ndarray) -> str:
    """Compact hash of parchment ink so the bot can detect line changes."""
    white, orange, dark = ink_layers(bgr)
    combo = cv2.bitwise_or(orange, cv2.bitwise_or(white, dark))
    if int((combo > 0).sum()) < 30:
        return "empty"
    small = cv2.resize(combo, (48, 12), interpolation=cv2.INTER_AREA)
    # 2-bit quantize → stable across tiny shimmer
    bits = (small > 40).astype(np.uint8)
    wp = int((white > 0).sum())
    op = int((orange > 0).sum())
    return f"{wp // 40}:{op // 40}:{hash(bits.tobytes()) & 0xFFFFFFFF:08x}"


def typed_remain_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Typed = white outline + dark fill away from orange; remain = orange."""
    white, orange, dark = ink_layers(bgr)
    near_orange = cv2.dilate(orange, np.ones((3, 3), np.uint8), iterations=1)
    # Typed letters: outline and/or dark body that isn't still-orange remain
    typed = cv2.bitwise_or(white, dark)
    typed = cv2.bitwise_and(typed, cv2.bitwise_not(near_orange))
    remain = cv2.bitwise_and(orange, cv2.bitwise_not(white))
    return typed, remain


def ink_stats(bgr: np.ndarray) -> tuple[int, int, float]:
    """typed_px, orange_px, typed_frac (dark+outline vs orange remain)."""
    typed, orange = typed_remain_masks(bgr)
    tp = int((typed > 0).sum())
    op = int((orange > 0).sum())
    frac = (tp / max(1, tp + op)) if (tp + op) else 0.0
    return tp, op, frac
