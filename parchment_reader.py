"""Read the battle parchment straight from screen pixels.

The parchment uses the game's own sprite font (spr_font_alagard) drawn at an
integer scale, in a fixed palette:

  parchment   BGR (184,200,216)       guide lines (136,200,216) / (104,168,200)
  untyped     brown  (40,88,152)
  typed       navy   (72,40,40) with a white (248,248,248) outline

Every letter is segmented by column, classified typed/untyped by its body
colour, and read by comparing it pixel-for-pixel with the font sprites, so the
bot knows the exact text and the exact typed position without guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from glyph_matcher import GlyphPack, load_primary_pack

# Sprite frames that are placeholders in the extracted pack (all identical boxes)
_JUNK_GLYPHS = set("*[]#^;|<>{}_")
# Characters drawn on the parchment that are never typed
_UNTYPED_CHARS = set(".")


@dataclass
class Glyph:
    x1: int
    x2: int
    y1: int
    y2: int
    typed: bool
    char: str = "?"
    conf: float = 0.0
    alts: list[str] = field(default_factory=list)


@dataclass
class ParchmentLine:
    words: list[list[Glyph]]
    unit: float

    @property
    def letters(self) -> list[Glyph]:
        return [g for w in self.words for g in w]

    @property
    def typed_count(self) -> int:
        """Leading run of typed letters (the game types strictly left to right)."""
        n = 0
        for g in self.letters:
            if not g.typed:
                break
            n += 1
        return n

    @property
    def total(self) -> int:
        return len(self.letters)

    @property
    def text(self) -> str:
        return " ".join("".join(g.char for g in w) for w in self.words)

    @property
    def min_conf(self) -> float:
        return min((g.conf for g in self.letters), default=0.0)

    def remaining(self, start: int | None = None) -> str:
        """Text still to type from letter index `start` (default: screen progress)."""
        start = self.typed_count if start is None else start
        out = ""
        idx = 0
        for word in self.words:
            chunk = "".join(g.char for i, g in enumerate(word) if idx + i >= start)
            idx += len(word)
            if chunk:
                out = f"{out} {chunk}" if out else chunk
        return out

    def at_word_start(self, index: int) -> bool:
        idx = 0
        for word in self.words:
            if index == idx:
                return index > 0
            idx += len(word)
        return False

    def describe(self) -> str:
        marks = []
        for w in self.words:
            marks.append("".join(g.char.upper() if g.typed else g.char.lower() for g in w))
        return " ".join(marks)


def color_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (parchment, brown, navy, white)."""
    B, G, R = [c.astype(np.int16) for c in cv2.split(bgr)]
    parch = (R >= 195) & (G >= 175) & (B >= 120) & (R - B >= 15) & (R - B <= 100) & (R - G <= 45)
    brown = (
        (R >= 110) & (R <= 210) & (G >= 45) & (G <= 130) & (B <= 100)
        & (R - G >= 35) & (G - B >= 12)
    )
    navy = (B >= 40) & (B <= 130) & (R <= 90) & (G <= 90) & (B - R >= 15)
    white = (R >= 225) & (G >= 225) & (B >= 225)
    return parch, brown, navy, white


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks] + 1, [idx[-1] + 1]))
    return list(zip(starts.tolist(), ends.tolist()))


def _parchment_rows(parch: np.ndarray) -> tuple[int, int] | None:
    """Largest contiguous block of rows that are mostly parchment."""
    frac = parch.mean(axis=1)
    rows = _runs(frac > 0.25)
    if not rows:
        return None
    y1, y2 = max(rows, key=lambda r: r[1] - r[0])
    if (y2 - y1) < max(8, parch.shape[0] * 0.12):
        return None
    return y1, y2


_unit_cache: dict[int, float] = {}


def _estimate_unit(band: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    """Font pixel size in screen pixels (6 at 3840x2160).

    The tallest letter is a whole number of font pixels (8..14); pick the
    count whose scale makes the letters match the font sprites best.
    Cached per tallest-letter height since the scale only changes with window size.
    """
    hmax = max(y2 - y1 for _a, _b, y1, y2 in boxes)
    if hmax in _unit_cache:
        return _unit_cache[hmax]
    sample = sorted(boxes, key=lambda bx: bx[1] - bx[0], reverse=True)[:8]
    best_unit, best_score = hmax / 11.0, -1.0
    for n in range(7, 15):
        unit = hmax / n
        if unit < 1.0:
            continue
        scores = [
            _read_glyph(band[y1:y2, a:b], unit)[1] for a, b, y1, y2 in sample
        ]
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best_unit, best_score = unit, score
    if best_score >= 0.6:
        _unit_cache[hmax] = best_unit
    return best_unit


@lru_cache(maxsize=1)
def _template_index() -> dict[tuple[int, int], list[tuple[str, np.ndarray]]]:
    pack: GlyphPack | None = load_primary_pack()
    index: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
    if pack is None:
        return index
    for ch, t in pack.templates.items():
        if ch in _JUNK_GLYPHS:
            continue
        tb = t > 0
        ys, xs = np.nonzero(tb)
        if ys.size == 0:
            continue
        tb = tb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        index.setdefault(tb.shape, []).append((ch, tb))
    return index


def _read_glyph(crop: np.ndarray, unit: float) -> tuple[str, float, list[str]]:
    """Match a tight binary crop of one letter against font sprites."""
    gh = max(1, int(round(crop.shape[0] / unit)))
    gw = max(1, int(round(crop.shape[1] / unit)))
    small = cv2.resize(crop.astype(np.uint8) * 255, (gw, gh), interpolation=cv2.INTER_AREA) > 127
    index = _template_index()
    scored: list[tuple[float, str]] = []
    for dh in (0, -1, 1):
        for dw in (0, -1, 1):
            cands = index.get((gh + dh, gw + dw))
            if not cands:
                continue
            probe = small
            if dh or dw:
                probe = cv2.resize(
                    crop.astype(np.uint8) * 255, (gw + dw, gh + dh), interpolation=cv2.INTER_AREA
                ) > 127
            penalty = 0.08 * (abs(dh) + abs(dw))
            for ch, tb in cands:
                inter = np.logical_and(tb, probe).sum()
                union = np.logical_or(tb, probe).sum()
                scored.append((inter / max(1, union) - penalty, ch))
    if not scored:
        return "?", 0.0, []
    scored.sort(reverse=True)
    best_score, best_ch = scored[0]
    alts = [ch for s, ch in scored[1:4] if s >= best_score - 0.05 and ch != best_ch]
    return best_ch, float(best_score), alts


def read_parchment(bgr: np.ndarray) -> ParchmentLine | None:
    """Segment, classify and read the parchment line. None = no parchment / no text."""
    if bgr is None or bgr.size == 0:
        return None
    parch, brown, navy, white = color_masks(bgr)
    block = _parchment_rows(parch)
    if block is None:
        return None
    py1, py2 = block
    if float(parch[py1:py2].mean()) < 0.35:
        return None

    body = (brown | navy)[py1:py2]
    if int(body.sum()) < 20:
        return None
    # Drop speckles (anti-aliasing / sparkles) before measuring geometry
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_min = max(3, int(np.percentile(areas, 90) * 0.02))
        good = np.zeros(n, bool)
        good[1:] = areas >= keep_min
        body = good[labels]
    if int(body.sum()) < 20:
        return None

    # Text band: rows holding ink, merged across small gaps (i-dots, accents)
    row_runs = _runs(body.sum(axis=1) > 0)
    if not row_runs:
        return None
    merged: list[list[int]] = [list(row_runs[0])]
    for a, b in row_runs[1:]:
        if a - merged[-1][1] <= max(4, (py2 - py1) // 12):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    by1, by2 = max(merged, key=lambda r: int(body[r[0] : r[1]].sum()))
    band = body[by1:by2]

    col_runs = _runs(band.any(axis=0))
    if not col_runs:
        return None
    boxes: list[tuple[int, int, int, int]] = []
    for a, b in col_runs:
        ys = np.flatnonzero(band[:, a:b].any(axis=1))
        boxes.append((a, b, int(ys[0]), int(ys[-1]) + 1))
    unit = _estimate_unit(band, boxes)
    hmax = max(y2 - y1 for _a, _b, y1, y2 in boxes)

    navy_b = navy[py1:py2][by1:by2]
    brown_b = brown[py1:py2][by1:by2]

    glyphs: list[Glyph] = []
    for (a, b, y1, y2) in boxes:
        # Too thin to be a letter (stray column of anti-aliasing)
        if (b - a) < unit * 0.6 and (y2 - y1) < unit * 0.6:
            continue
        crop = band[y1:y2, a:b]
        ch, conf, alts = _read_glyph(crop, unit)
        tiny = (y2 - y1) <= hmax * 0.3 and (b - a) <= unit * 3.5
        if tiny and ch not in ("'", ",", "-"):
            ch = "."
        if ch in _UNTYPED_CHARS:
            continue
        n_navy = int(navy_b[y1:y2, a:b].sum())
        n_brown = int(brown_b[y1:y2, a:b].sum())
        glyphs.append(
            Glyph(
                x1=a, x2=b, y1=y1 + by1 + py1, y2=y2 + by1 + py1,
                typed=n_navy > n_brown, char=ch, conf=conf, alts=alts,
            )
        )
    if not glyphs:
        return None

    # Word breaks: letter gaps are <= ~5 font px, a space is ~15
    words: list[list[Glyph]] = [[glyphs[0]]]
    for prev, g in zip(glyphs, glyphs[1:]):
        if g.x1 - prev.x2 > unit * 8:
            words.append([g])
        else:
            words[-1].append(g)

    _fix_l_vs_i(words)
    return ParchmentLine(words=words, unit=unit)


def _fix_l_vs_i(words: list[list[Glyph]]) -> None:
    """'l' and 'I' share one sprite: capital I starts a word, l is inside one."""
    for word in words:
        for i, g in enumerate(word):
            if g.char in ("l", "I"):
                g.char = "I" if i == 0 else "l"
                other = "l" if g.char == "I" else "I"
                if other not in g.alts:
                    g.alts.insert(0, other)


def build_vocab(words: list[str]) -> dict[int, dict[str, str]]:
    """Index vocabulary by length: {len: {lowercase: original}}."""
    vocab: dict[int, dict[str, str]] = {}
    for w in words:
        w = w.strip()
        if w and w.isalpha():
            vocab.setdefault(len(w), {}).setdefault(w.lower(), w)
    return vocab


def correct_with_vocab(
    line: ParchmentLine, vocab: dict[int, dict[str, str]], min_conf: float = 0.85
) -> None:
    """Fix only low-confidence letters, using a same-length vocabulary word."""
    from rapidfuzz import fuzz, process

    for word in line.words:
        weak = [g.conf < min_conf or g.char == "?" for g in word]
        if not any(weak):
            continue
        cands = vocab.get(len(word))
        if not cands:
            continue
        raw = "".join(g.char for g in word).lower()
        hit = process.extractOne(raw, list(cands), scorer=fuzz.ratio)
        if not hit or hit[1] < 60:
            continue
        fixed = cands[hit[0]]
        for g, ch, is_weak in zip(word, fixed, weak):
            if is_weak:
                g.char = ch


def debug_overlay(bgr: np.ndarray, line: ParchmentLine | None) -> np.ndarray:
    """Draw letter boxes: green = typed, red = untyped."""
    out = bgr.copy()
    if line is None:
        return out
    for g in line.letters:
        color = (0, 200, 0) if g.typed else (0, 0, 255)
        cv2.rectangle(out, (g.x1, g.y1), (g.x2 - 1, g.y2 - 1), color, 2)
    return out
