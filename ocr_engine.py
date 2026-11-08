"""Parchment OCR + dictionary correction for Textorcist.

Battle text only: beige parchment ROI. Dialogue on black backgrounds is ignored.
Glyph packs verify / re-rank phrase candidates against the game's own fonts.
Typed vs remaining ink (black/blue vs brown) drives interrupt resume.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz

from glyph_matcher import (
    best_phrase_by_glyphs,
    is_battle_parchment,
    load_primary_pack,
    split_phrase_by_fraction,
    typed_fraction_from_masks,
)
from glyph_reader import (
    ProgressSplit,
    ink_fingerprint,
    ink_layers,
    ink_stats,
    split_progress_by_glyphs,
    typed_remain_masks,
)


def _layout_pick_phrase(
    bgr,
    candidates: list[str],
    raw_hint: str = "",
) -> tuple[str, ProgressSplit] | None:
    """Pick the phrase whose glyph-layout progress matches screen ink."""
    best: tuple[float, str, ProgressSplit] | None = None
    seen: set[str] = set()
    raw_c = compact(raw_hint)
    for phrase in candidates:
        phrase = strip_ellipsis(phrase)
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        prog = split_progress_by_glyphs(bgr, phrase)
        words = phrase.split()
        if not words:
            continue
        score = 0.0
        pc = compact(phrase)
        if raw_c and len(raw_c) >= 8:
            score += float(fuzz.partial_ratio(raw_c, pc)) * 0.55
            # Prefer phrases whose length fits OCR (short hop vs long prayer)
            len_delta = abs(len(pc) - len(raw_c))
            score += max(0.0, 25.0 - len_delta * 1.2)
            if len(pc) > len(raw_c) + max(8, len(raw_c) // 2):
                score -= 35.0
            for anchor in ("witness", "cross", "lord", "flee", "archangel", "mercy", "hope", "have"):
                if anchor in raw_c and anchor not in pc:
                    score -= 40.0
                if anchor in raw_c and anchor in pc:
                    score += 14.0
        if prog.method in {"glyph_layout", "column_cut"}:
            score += 15.0
        if prog.remaining and prog.already:
            score += 25.0
        elif prog.remaining and not prog.already and prog.typed_frac < 0.2:
            score += 30.0  # fresh short phrase still fully brown
        elif prog.already and not prog.remaining and prog.typed_frac > 0.55:
            score += 12.0
        if phrase.lower().endswith("flee ye hostiles"):
            score -= 20.0
        elif phrase.lower().endswith(" flee") and "flee" in raw_c:
            score += 10.0
        if best is None or score > best[0]:
            best = (score, phrase, prog)
    if best is None:
        return None
    return best[1], best[2]


@dataclass
class OcrResult:
    raw: str
    matched: str
    score: float
    remaining: str = ""
    already_typed: str = ""
    parchment_present: bool = True
    progress_reset: bool = False
    ink_fp: str = ""
    typed_frac: float = 0.0


_I_LIKE = set("il1|!")


def parchment_present(bgr: np.ndarray) -> bool:
    """True when the beige combat typing strip is on screen (book held).

    Explicitly rejects dialogue / story text drawn over black backgrounds.
    """
    if not is_battle_parchment(bgr):
        return False
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    dark = int((lab[:, :, 0] < 120).sum())
    area = bgr.shape[0] * bgr.shape[1]
    # Need some ink, but not a full dark scene
    return 80 < dark < area * 0.45


def _clean_dark_mask(dark_u8: np.ndarray) -> np.ndarray:
    h, w = dark_u8.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    horizontal = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, h_kernel)
    dark_u8 = cv2.subtract(dark_u8, horizontal)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 2)))
    vertical = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, v_kernel)
    dark_u8 = cv2.subtract(dark_u8, vertical)
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return dark_u8


def ink_mask(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    dark = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return _clean_dark_mask(dark)


def split_typed_remaining_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split parchment ink into:
      typed     = already typed (black or blue/purple)
      remaining = still to type (warm brown)
      all_ink   = typed | remaining  (excludes grey UI chrome)
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    dark = _clean_dark_mask(
        cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    )
    dark_b = dark > 0
    B, G, R = cv2.split(bgr)
    R16, G16, B16 = R.astype(np.int16), G.astype(np.int16), B.astype(np.int16)
    b_lab = lab[:, :, 2].astype(np.int16)
    # Untyped combat ink: warm brown/orange on parchment
    brown = dark_b & (R16 > G16 + 8) & ((R16 - B16) > 45) & (b_lab > 140)
    # Typed combat ink: near-black OR cool blue/purple (not grey ornaments)
    near_black = dark_b & (R16 < 70) & (G16 < 70) & (B16 < 70)
    blue = dark_b & (B16 > R16 + 25) & (B16 > G16 + 15) & (B16 > 70)
    typed = near_black | blue
    all_ink = brown | typed
    return (
        typed.astype(np.uint8) * 255,
        brown.astype(np.uint8) * 255,
        all_ink.astype(np.uint8) * 255,
    )


def restrict_masks_to_text_span(
    typed: np.ndarray,
    remain: np.ndarray,
    all_ink: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ignore left/right parchment chrome; keep the horizontal span of brown text."""
    if remain is None or int((remain > 0).sum()) < 40:
        return typed, remain, all_ink
    ys, xs = np.where(remain > 0)
    if len(xs) < 10:
        return typed, remain, all_ink
    pad = max(8, int((xs.max() - xs.min()) * 0.04))
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(remain.shape[1], int(xs.max()) + pad + 1)
    # Also drop extreme top/bottom ornament rows outside brown band
    ypad = 6
    y1 = max(0, int(ys.min()) - ypad)
    y2 = min(remain.shape[0], int(ys.max()) + ypad + 1)
    gate = np.zeros(remain.shape, np.uint8)
    gate[y1:y2, x1:x2] = 255
    typed = cv2.bitwise_and(typed, gate)
    remain = cv2.bitwise_and(remain, gate)
    all_ink = cv2.bitwise_and(all_ink, gate)
    return typed, remain, all_ink


def content_crop(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return None
    if len(xs) < mask.size * 0.0015:
        return None

    pad = 8
    x1, x2 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + pad)
    y1, y2 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + pad)
    if (x2 - x1) > bgr.shape[1] * 0.95 and (y2 - y1) < bgr.shape[0] * 0.35:
        if len(xs) < 80:
            return None
    return bgr[y1:y2, x1:x2]


def mask_to_bw(mask: np.ndarray) -> np.ndarray | None:
    if mask is None or mask.size == 0 or int(mask.sum()) < 40 * 255:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    pad = 6
    y1, y2 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    crop = mask[y1:y2, x1:x2]
    crop = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    crop = cv2.dilate(crop, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.bitwise_not(crop)


def preprocess(bgr: np.ndarray) -> np.ndarray | None:
    mask = ink_mask(bgr)
    cropped = content_crop(bgr, mask)
    if cropped is None:
        return None
    mask_c = ink_mask(cropped)
    return mask_to_bw(mask_c)


def strip_ellipsis(text: str) -> str:
    text = re.sub(r"^\s*\.{2,}\s*", "", text)
    text = re.sub(r"\s*\.{2,}\s*", " ", text)
    text = re.sub(r"\s*\.+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("|", "I")
    text = re.sub(r"[^A-Za-z0-9'!?.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return strip_ellipsis(text)


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_OCR_ANCHORS = (
    "witness",
    "flee",
    "away",
    "hostiles",
    "maythy",
    "sancte",
    "michael",
    "archangel",
    "amen",
    "lord",
    "cross",
    "forwe",
    "havehope",
    "mercy",
)


def strip_leading_ocr_junk(raw_c: str) -> list[str]:
    """Drop chrome/noise before the real combat phrase (e.g. 'coee me' + Witness…)."""
    out = {raw_c}
    if len(raw_c) < 6:
        return [raw_c]
    for anchor in _OCR_ANCHORS:
        idx = raw_c.find(anchor)
        if idx > 0:
            out.add(raw_c[idx:])
    # Also try cutting up to ~12 leading junk letters
    for n in range(1, min(13, max(1, len(raw_c) - 8))):
        out.add(raw_c[n:])
    return [v for v in out if len(v) >= 4]


def ocr_core(raw_c: str) -> str:
    """Best junk-stripped OCR core (cut at earliest combat anchor)."""
    if not raw_c:
        return ""
    best_idx: int | None = None
    for anchor in _OCR_ANCHORS:
        idx = raw_c.find(anchor)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
    if best_idx is not None and best_idx > 0:
        return raw_c[best_idx:]
    return raw_c


def ocr_variants(raw_c: str) -> list[str]:
    variants = {raw_c}
    # Common single-word misreads
    singles = {
        "tlee": "flee",
        "piew": "flee",
        "fiee": "flee",
        "filee": "flee",
        "file": "flee",
        "flee": "flee",
        "tlea": "flee",
        "flec": "flee",
        "awad": "away",
        "awai": "away",
        "avvay": "away",
        "wagu": "away",
        "hostitles": "hostiles",
        "wostiles": "hostiles",
        "hostiles": "hostiles",
    }
    if raw_c in singles:
        variants.add(singles[raw_c])
    for src, dst in singles.items():
        if src in raw_c:
            variants.add(raw_c.replace(src, dst))
    for src in ("wessel", "wesset", "messed", "vessed", "vesse1", "aechangel", "archanget"):
        if src in raw_c:
            if src in ("aechangel", "archanget"):
                variants.add(raw_c.replace(src, "archangel"))
            else:
                variants.add(raw_c.replace(src, "vessel"))
    variants.add(raw_c.replace("vv", "w"))
    # OfF / offthe (double-f) → of / ofthe
    if "offthe" in raw_c:
        variants.add(raw_c.replace("offthe", "ofthe"))
    if "crossof" in raw_c and "crossoff" in raw_c.replace("ofthe", "offthe"):
        pass
    variants.add(raw_c.replace("crossoff", "crossof"))
    if "tlee" in raw_c:
        variants.add(raw_c.replace("tlee", "flee"))
    if "filee" in raw_c:
        variants.add(raw_c.replace("filee", "flee"))
    if "flatrd" in raw_c:
        variants.add(raw_c.replace("flatrd", "lord"))
    for cleaned in strip_leading_ocr_junk(raw_c):
        variants.add(cleaned)
        if "offthe" in cleaned:
            variants.add(cleaned.replace("offthe", "ofthe"))
        if "filee" in cleaned:
            variants.add(cleaned.replace("filee", "flee"))
        if cleaned.endswith("filee"):
            variants.add(cleaned[: -len("filee")] + "flee")
        if cleaned.endswith("file"):
            variants.add(cleaned[: -len("file")] + "flee")
    for src, dst in (("omen", "amen"), ("anen", "amen"), ("ameii", "amen"), ("hitie", "amen")):
        if raw_c.endswith(src):
            variants.add(raw_c[: -len(src)] + dst)
        variants.add(raw_c.replace(src, dst))
    if len(raw_c) > 5 and raw_c[0] in "o0ac":
        stripped = raw_c[1:]
        variants.add(stripped)
        for src in ("wessel", "wesset", "messed", "vessed"):
            if src in stripped:
                variants.add(stripped.replace(src, "vessel"))
        if stripped.endswith("omen"):
            base = stripped[:-4].replace("messed", "vessel").replace("wesset", "vessel")
            variants.add(base + "amen")
            variants.add(stripped[:-4] + "amen")
    if raw_c and raw_c[-1] in _I_LIKE:
        variants.add(raw_c[:-1] + "i")
    if len(raw_c) > 4 and raw_c[-1] in "aeo":
        variants.add(raw_c[:-1])
        if raw_c[-2] in _I_LIKE:
            variants.add(raw_c[:-2] + "i")
    m = re.search(r"^(.*i)[a-z]{1,3}$", raw_c)
    if m:
        variants.add(m.group(1))
    return [v for v in variants if len(v) >= 2]


def looks_like_garbage(raw: str) -> bool:
    if not raw or len(compact(raw)) < 2:
        return True
    tokens = raw.split()
    if len(tokens) >= 6 and sum(1 for t in tokens if len(t) <= 2) >= len(tokens) * 0.6:
        return True
    if len(set(tokens)) <= 3 and len(tokens) >= 5:
        return True
    if compact(raw) in {"e", "ee", "eee", "s", "ss", "sss"}:
        return True
    return False


def tesseract_bw(image: np.ndarray) -> str:
    configs = [
        (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'!?.- "
        ),
        (
            "--oem 3 --psm 6 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'!?.- "
        ),
    ]
    best = ""
    for config in configs:
        raw = normalize_text(pytesseract.image_to_string(image, config=config))
        if not raw:
            continue
        if not looks_like_garbage(raw) and len(compact(raw)) > len(compact(best)):
            best = raw
        elif not best:
            best = raw
    return best


def run_tesseract(bgr: np.ndarray) -> str:
    image = preprocess(bgr)
    if image is None:
        return ""
    return tesseract_bw(image)


def run_tesseract_mask(mask: np.ndarray) -> str:
    image = mask_to_bw(mask)
    if image is None:
        return ""
    return tesseract_bw(image)


def _span_score(raw_c: str, span_c: str) -> float:
    if not raw_c or not span_c:
        return 0.0
    max_delta = max(3, len(raw_c) // 5)
    if abs(len(span_c) - len(raw_c)) > max_delta:
        return 0.0
    return float(fuzz.ratio(raw_c, span_c))


def best_span_from_lines(raw: str, raw_c: str, lines: list[str]) -> tuple[str, float]:
    best_span, best_score = "", 0.0
    variants = ocr_variants(raw_c)
    has_standalone_i = bool(
        re.search(r"(?<![A-Za-z])(?:I|l|1|\||Lt|tl)(?![A-Za-z])", raw)
    )
    wants_trailing_i = (
        has_standalone_i
        or any(v.endswith("i") for v in variants)
        or raw_c[-1:] in _I_LIKE
        or "i" in raw_c[-5:]
        or bool(re.search(r"l[t1]?a?$", raw_c))
    )

    for line in lines:
        words = line.split()
        n = len(words)
        compacts = [compact(w) for w in words]
        for i in range(n):
            acc = ""
            for j in range(i, n):
                acc += compacts[j]
                span = " ".join(words[i : j + 1])
                for variant in variants:
                    score = _span_score(variant, acc)
                    if score <= 0:
                        continue
                    if wants_trailing_i and words[j] == "I":
                        score += 6.0
                    score += min(len(acc), 48) * 0.35
                    if score > best_score:
                        best_score, best_span = score, span

    if best_span and wants_trailing_i and not best_span.endswith(" I"):
        for line in lines:
            if best_span in line:
                words = line.split()
                span_words = best_span.split()
                for idx in range(len(words) - len(span_words) + 1):
                    if words[idx : idx + len(span_words)] == span_words:
                        end = idx + len(span_words)
                        if end < len(words) and words[end] == "I":
                            extended = best_span + " I"
                            if len(raw_c) >= len(compact(extended)) - 3:
                                best_span = extended
                                best_score = min(best_score + 4.0, 100.0)
                        break

    return best_span, min(best_score, 100.0)


def best_phrase_match(
    raw: str,
    phrases: list[str],
    min_score: float,
    lines: list[str] | None = None,
) -> tuple[str, float]:
    raw = strip_ellipsis(raw)
    if not raw or looks_like_garbage(raw):
        return "", 0.0

    raw_c = compact(raw)
    if len(raw_c) < 2:
        return "", 0.0

    best_matched, best_score = "", 0.0

    if lines:
        span, score = best_span_from_lines(raw, raw_c, lines)
        if span and score > best_score:
            best_matched, best_score = span, score

    max_delta = max(3, len(raw_c) // 5)
    variants = ocr_variants(raw_c)
    for p in phrases:
        pc = compact(p)
        if abs(len(pc) - len(raw_c)) > max_delta:
            continue
        for variant in variants:
            score = float(fuzz.ratio(variant, pc))
            if score > best_score or (
                score == best_score and len(pc) > len(compact(best_matched))
            ):
                best_matched, best_score = p, score

    needed = min_score
    if len(raw_c) <= 3:
        needed = max(min_score, 92.0)
    elif len(raw_c) <= 5:
        needed = max(min_score, 85.0)

    if best_score < needed or not best_matched:
        return "", best_score
    return strip_ellipsis(best_matched), best_score


def best_suffix_of(phrase: str, remaining_raw: str, min_score: float) -> str:
    """Map brown-ink OCR onto a suffix of the known phrase (resume mid-word)."""
    remaining_raw = strip_ellipsis(remaining_raw)
    if not phrase or not remaining_raw or looks_like_garbage(remaining_raw):
        return ""

    rem_c = compact(remaining_raw)
    if len(rem_c) < 1:
        return ""

    variants = ocr_variants(rem_c)
    phrase_c = compact(phrase)
    best_suffix, best_score = "", 0.0

    # Character-aligned suffixes (resume inside a word, e.g. 'hael...')
    for i in range(len(phrase_c)):
        suffix_c = phrase_c[i:]
        if abs(len(suffix_c) - len(rem_c)) > max(3, len(rem_c) // 4 + 2):
            continue
        for variant in variants:
            score = float(fuzz.ratio(variant, suffix_c))
            if score > best_score:
                best_score = score
                # Map compact index back to original phrase string
                seen = 0
                start_idx = 0
                for idx, ch in enumerate(phrase):
                    if ch.isalnum():
                        if seen == i:
                            start_idx = idx
                            break
                        seen += 1
                best_suffix = phrase[start_idx:]

    if best_score >= max(min_score - 5, 60):
        return best_suffix.lstrip()
    return ""


def compact_leftover(raw_c: str, matched_c: str) -> str:
    """Letters in OCR after the matched phrase (e.g. '...LordFlee' → 'flee')."""
    if not raw_c or not matched_c:
        return ""
    best_end, best = min(len(raw_c), len(matched_c)), -1.0
    lo = max(1, len(matched_c) - 5)
    hi = min(len(raw_c), len(matched_c) + 12)
    for end in range(lo, hi + 1):
        score = float(fuzz.ratio(raw_c[:end], matched_c))
        if score > best:
            best, best_end = score, end
    if best < 55:
        return ""
    return raw_c[best_end:]


def _length_ok(raw_c: str, match_c: str) -> bool:
    """Reject under-matching (long OCR → short word like Hostiles)."""
    if not match_c:
        return False
    # OCR may have a bit of junk, but not almost 2x the match
    if len(raw_c) > len(match_c) + max(3, len(match_c) // 3):
        return False
    if len(match_c) > len(raw_c) + max(3, len(raw_c) // 3):
        return False
    return True


def match_vocab_word(raw: str, words: list[str], min_score: float) -> tuple[str, float]:
    """Match short leftover / brown OCR to a single combat hop-word."""
    raw = strip_ellipsis(raw)
    if not raw or not words:
        return "", 0.0
    raw_c = compact(raw)
    if len(raw_c) < 2:
        return "", 0.0
    variants = list(ocr_variants(raw_c))
    for v in list(variants):
        for n in range(3, min(9, len(v) + 1)):
            variants.append(v[-n:])

    best_w, best_s = "", 0.0
    for w in words:
        wc = compact(w)
        if not any(_length_ok(v, wc) for v in variants):
            continue
        for variant in variants:
            if not _length_ok(variant, wc):
                continue
            score = float(fuzz.ratio(variant, wc))
            if score > best_s:
                best_w, best_s = w, score
    needed = max(min_score, 70.0) if len(raw_c) <= 5 else min_score
    # Long OCR must not collapse to a short word
    if best_w and not _length_ok(raw_c, compact(best_w)):
        return "", best_s
    if best_s < needed:
        return "", best_s
    return best_w, best_s


def match_hop_phrase(
    raw: str,
    hop_phrases: list[str],
    words: list[str],
    min_score: float,
) -> tuple[str, float]:
    """Prefer multi-word hop sequences (Away Ye Hostiles) over a single word."""
    raw = strip_ellipsis(raw)
    if not raw:
        return "", 0.0
    raw_c = compact(raw)
    if len(raw_c) < 3:
        return match_vocab_word(raw, words, min_score)

    variants: list[str] = []
    # Leading OCR junk is common ("lcces" + real text) — try stripping prefixes
    # Only light leading-junk strip (keep Away / first word)
    for start in range(0, min(5, max(1, len(raw_c) - 4))):
        variants.extend(ocr_variants(raw_c[start:]))
    seen_v: set[str] = set()
    uniq_v: list[str] = []
    for v in variants:
        if v not in seen_v:
            seen_v.add(v)
            uniq_v.append(v)
    variants = uniq_v

    core_c = ocr_core(raw_c)
    # Prefer length-similar hops for short OCR (For We Have Hope), not longest-first
    best_p, best_s = "", 0.0
    candidates = list(hop_phrases) if hop_phrases else []
    candidates.sort(
        key=lambda p: (abs(len(compact(p)) - len(core_c)), -len(compact(p)))
    )
    for p in candidates:
        pc = compact(p)
        for variant in variants:
            # Variant must be about as long as the phrase (don't match Hostiles
            # by chopping OCR down to the last word)
            if abs(len(variant) - len(pc)) > max(2, len(pc) // 4):
                continue
            if len(variant) < max(4, int(0.75 * len(pc))):
                continue
            score = float(fuzz.ratio(variant, pc))
            # Coverage vs junk-stripped core (ignore leading chrome)
            cover = len(variant) / max(len(core_c), 1)
            score = score * (0.55 + 0.45 * min(cover, 1.0))
            score += min(len(pc), 48) * 0.35
            if _length_ok(core_c, pc):
                score += 8.0
            if score > best_s:
                best_p, best_s = p, score

    if not best_p or best_s < min_score:
        w, ws = match_vocab_word(raw, words, min_score)
        if w and ws > best_s and (
            _length_ok(core_c, compact(w)) or _length_ok(raw_c, compact(w))
        ):
            return w, ws
        if best_s < min_score:
            return "", best_s
    if best_p and best_s >= min_score:
        return best_p, min(best_s, 100.0)
    return "", best_s


def remaining_from_progress(
    phrase: str,
    typed_raw: str,
    remaining_raw: str,
    black_px: int,
    red_px: int,
    min_score: float,
) -> tuple[str, str, bool]:
    """
    Returns (already_typed, remaining, progress_reset).
    Uses ink colors as source of truth after interruptions / book pickup.
    """
    if not phrase:
        return "", "", False

    phrase_c = compact(phrase)
    total_ink = black_px + red_px
    black_frac = (black_px / total_ink) if total_ink else 0.0

    # No / negligible typed ink => fresh line (chrome / gothic edges often
    # look "dark" and must not count as progress)
    if black_px < 80 and red_px > 80:
        return "", phrase, True
    if black_frac < 0.18 and red_px > 100:
        return "", phrase, True

    # Fully typed
    if red_px < 40 and black_px > 80:
        return phrase, "", False

    rem = best_suffix_of(phrase, remaining_raw, min_score) if remaining_raw else ""
    if rem and len(compact(rem)) >= max(2, int(0.35 * len(phrase_c))):
        # Derive typed prefix from phrase
        if phrase.endswith(rem):
            already = phrase[: -len(rem)]
        else:
            # rem may be mid-string equal to suffix via compact; use compact split
            rem_c = compact(rem)
            if phrase_c.endswith(rem_c):
                keep = len(phrase_c) - len(rem_c)
                seen = 0
                cut = 0
                for idx, ch in enumerate(phrase):
                    if ch.isalnum():
                        if seen == keep:
                            cut = idx
                            break
                        seen += 1
                already = phrase[:cut]
            else:
                already = ""
        return already, rem, False

    # Fallback: typed OCR as prefix (ignore 1-letter noise from ornaments)
    typed_raw = strip_ellipsis(typed_raw)
    if (
        typed_raw
        and not looks_like_garbage(typed_raw)
        and len(compact(typed_raw)) >= 2
        and black_frac >= 0.20
    ):
        typed_c = compact(typed_raw)
        variants = ocr_variants(typed_c)
        best_i, best_score = 0, 0.0
        for i in range(1, len(phrase_c) + 1):
            prefix = phrase_c[:i]
            for variant in variants:
                score = float(fuzz.ratio(variant, prefix))
                if score > best_score:
                    best_score, best_i = score, i
        if best_score >= max(min_score - 5, 60):
            seen = 0
            cut = len(phrase)
            for idx, ch in enumerate(phrase):
                if ch.isalnum():
                    if seen == best_i:
                        cut = idx
                        break
                    seen += 1
            already = phrase[:cut]
            return already, phrase[cut:].lstrip(), False

    # Pixel-ratio fallback on compact letters
    total = black_px + red_px
    if total > 0:
        frac = black_px / total
        cut_letters = int(round(frac * len(phrase_c)))
        cut_letters = max(0, min(len(phrase_c), cut_letters))
        seen = 0
        cut = 0
        for idx, ch in enumerate(phrase):
            if ch.isalnum():
                if seen == cut_letters:
                    cut = idx
                    break
                seen += 1
            cut = idx + 1
        already = phrase[:cut]
        rem = phrase[cut:].lstrip()
        return already, rem, False

    return "", phrase, False


def _safe_tesseract_mask(mask: np.ndarray) -> str:
    try:
        return run_tesseract_mask(mask)
    except pytesseract.TesseractNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _safe_tesseract_bgr(bgr: np.ndarray) -> str:
    try:
        return run_tesseract(bgr)
    except pytesseract.TesseractNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        return ""


def space_ocr_with_words(raw: str, words: list[str] | None) -> str:
    """Turn 'ForWeHaveHope' into 'For We Have Hope' using the word list."""
    raw = strip_ellipsis(raw or "")
    raw_c = compact(raw)
    if not raw_c:
        return ""
    if " " in raw.strip() and len(raw.split()) >= 2:
        return re.sub(r"\s+", " ", raw).strip()
    if not words:
        return raw.strip()
    # Longest-first greedy tokenize
    lexicon = sorted({w for w in words if w and w[0].isalpha()}, key=len, reverse=True)
    compact_map = [(compact(w), w) for w in lexicon if len(compact(w)) >= 2]
    out: list[str] = []
    i = 0
    while i < len(raw_c):
        hit = None
        for wc, w in compact_map:
            if raw_c.startswith(wc, i):
                hit = w
                i += len(wc)
                break
        if hit is None:
            # take one letter and keep going
            out.append(raw_c[i])
            i += 1
        else:
            out.append(hit)
    # Merge single-letter crumbs into previous token when possible
    spaced: list[str] = []
    buf = ""
    for tok in out:
        if len(tok) == 1 and tok.islower():
            buf += tok
            continue
        if buf:
            spaced.append(buf)
            buf = ""
        spaced.append(tok)
    if buf:
        spaced.append(buf)
    return " ".join(spaced).strip()


def polish_screen_text(
    raw: str,
    words: list[str] | None,
    hop_phrases: list[str] | None = None,
    phrases: list[str] | None = None,
) -> str:
    """Prefer a known hop/phrase only when OCR is clearly that text; else space OCR."""
    raw = strip_ellipsis(raw or "").strip()
    if not raw or looks_like_garbage(raw):
        return ""
    raw_c = ocr_core(compact(raw))
    if len(raw_c) < 3:
        return ""
    # Exact-ish hop / short phrase wins for spacing
    for pool in (hop_phrases or [], phrases or []):
        best_p, best_s = "", 0.0
        for p in pool:
            pc = compact(p)
            if abs(len(pc) - len(raw_c)) > max(2, len(raw_c) // 5):
                continue
            sc = float(fuzz.ratio(raw_c, pc))
            if sc > best_s:
                best_p, best_s = p, sc
        if best_p and best_s >= 86:
            return strip_ellipsis(best_p)
    spaced = space_ocr_with_words(raw_c, words)
    return spaced or raw


def _normalize_ocr_compact(raw_c: str) -> str:
    s = (raw_c or "").lower()
    for src, dst in (
        ("filee", "flee"),
        ("fiee", "flee"),
        ("flec", "flee"),
        ("crossoff", "crossof"),
        ("lorad", "lord"),
        ("larad", "lord"),
        ("lhe", "the"),
    ):
        s = s.replace(src, dst)
    return s


def map_orange_to_phrase_suffix(phrase: str, orange_raw: str) -> tuple[str, str] | None:
    """Map orange-ink OCR onto a suffix of the known phrase (resume after book pickup)."""
    phrase = strip_ellipsis(phrase or "")
    if not phrase or not orange_raw:
        return None
    rem_c = _normalize_ocr_compact(ocr_core(compact(orange_raw)))
    if len(rem_c) < 3:
        rem_c = _normalize_ocr_compact(compact(orange_raw))
    if len(rem_c) < 3:
        return None
    hint_c = compact(phrase)

    # Exact compact substring (prefer rightmost = true remain)
    if rem_c in hint_c:
        idx = hint_c.rfind(rem_c)
        # Avoid matching a tiny mid-word fragment as the whole remain
        if len(rem_c) >= 4 or idx + len(rem_c) == len(hint_c):
            seen = 0
            cut = 0
            for i, ch in enumerate(phrase):
                if ch.isalnum():
                    if seen == idx:
                        cut = i
                        break
                    seen += 1
            already = phrase[:cut].rstrip()
            remain = phrase[cut:].lstrip()
            if remain:
                return already, remain

    # Fuzzy suffix of phrase vs orange OCR
    suf = best_suffix_of(phrase, orange_raw, 65)
    if not suf:
        # retry with normalized compact glued as fake raw
        suf = best_suffix_of(phrase, rem_c, 65)
    if suf and len(compact(suf)) >= 3:
        if phrase.endswith(suf):
            already = phrase[: -len(suf)].rstrip()
        else:
            sc = compact(suf)
            if hint_c.endswith(sc):
                keep = len(hint_c) - len(sc)
                seen = 0
                cut = 0
                for i, ch in enumerate(phrase):
                    if ch.isalnum():
                        if seen == keep:
                            cut = i
                            break
                        seen += 1
                already = phrase[:cut].rstrip()
                suf = phrase[cut:].lstrip()
            else:
                already = ""
        return already, suf
    return None


def read_screen_progress(
    bgr: np.ndarray,
    words: list[str] | None = None,
    hop_phrases: list[str] | None = None,
    phrases: list[str] | None = None,
    hint_phrase: str = "",
) -> tuple[str, str, float, str, str]:
    """Screenshot-first progress.

    Returns (already, remaining, typed_frac, ink_fp, raw_orange_ocr).
    Orange ink OCR is the remain source; with a hint phrase we map it to a suffix
    so book-drop resume continues after Lord instead of retyping Witness.
    """
    fp = ink_fingerprint(bgr)
    tp, op, typed_frac = ink_stats(bgr)
    typed_m, orange_m = typed_remain_masks(bgr)

    if op < 30 and tp > 80:
        done = (
            polish_screen_text(
                _safe_tesseract_mask(typed_m) or hint_phrase,
                words,
                hop_phrases,
                phrases,
            )
            or hint_phrase
        )
        return done, "", typed_frac, fp, ""

    raw_remain = _safe_tesseract_mask(orange_m) if op > 25 else ""
    raw_typed = _safe_tesseract_mask(typed_m) if tp > 40 else ""

    # Known phrase: always try orange OCR → suffix (handles book pickup mid-line)
    if hint_phrase:
        mapped = map_orange_to_phrase_suffix(hint_phrase, raw_remain) if raw_remain else None
        if mapped is not None:
            already, remain = mapped
            # Orange OCR much shorter than full phrase ⇒ mid-progress, trust it
            if compact(remain) != compact(hint_phrase) or typed_frac < 0.12:
                return already, remain, typed_frac, fp, raw_remain
        # Glyph/column cut on the known phrase
        prog = split_progress_by_glyphs(bgr, hint_phrase)
        if prog.remaining and (
            prog.method in {"column_cut", "glyph_layout", "all_remain", "all_typed"}
            or typed_frac >= 0.12
        ):
            return prog.already, prog.remaining, typed_frac, fp, raw_remain
        if mapped is not None:
            return mapped[0], mapped[1], typed_frac, fp, raw_remain

    remain = polish_screen_text(raw_remain, words, hop_phrases, phrases)
    already = polish_screen_text(raw_typed, words, hop_phrases, phrases)

    # Fresh line (little typed ink): full remain, no fake prefix
    if typed_frac < 0.12 and op > 60:
        if not remain and hint_phrase:
            remain = hint_phrase
        return "", remain or hint_phrase, typed_frac, fp, raw_remain

    if not remain and hint_phrase and typed_frac < 0.85:
        prog = split_progress_by_glyphs(bgr, hint_phrase)
        return prog.already, prog.remaining, typed_frac, fp, raw_remain

    return already or "", remain or "", typed_frac, fp, raw_remain


def _candidate_pool(
    phrases: list[str],
    lines: list[str] | None,
    hop_phrases: list[str] | None,
    words: list[str] | None,
    raw_hints: list[str],
    limit: int = 80,
) -> list[str]:
    """Build a shortlist of phrases to verify with glyph rendering."""
    pool: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        p = strip_ellipsis(p)
        if not p:
            return
        k = p.lower()
        if k in seen:
            return
        seen.add(k)
        pool.append(p)

    for raw in raw_hints:
        if not raw:
            continue
        hit, _sc = best_phrase_match(raw, phrases, min_score=50, lines=lines)
        if hit:
            add(hit)
        hop, _hs = match_hop_phrase(raw, hop_phrases or [], words or [], min_score=50)
        if hop:
            add(hop)

    # Length-similar phrases from OCR hints
    for raw in raw_hints:
        rc = compact(raw)
        if len(rc) < 3:
            continue
        for p in phrases:
            pc = compact(p)
            if abs(len(pc) - len(rc)) <= max(3, len(rc) // 4):
                add(p)
                if len(pool) >= limit:
                    return pool

    for p in (hop_phrases or [])[:40]:
        add(p)
    for w in (words or [])[:40]:
        add(w)
    return pool[:limit]


def recognize_locked(
    bgr: np.ndarray,
    phrase: str,
    words: list[str] | None = None,
    hop_phrases: list[str] | None = None,
    phrases: list[str] | None = None,
) -> OcrResult:
    """Fast path: re-read typed/remain from screen ink (orange = todo)."""
    if not parchment_present(bgr):
        return OcrResult(
            raw="",
            matched="",
            score=0.0,
            remaining="",
            already_typed="",
            parchment_present=False,
        )
    phrase = strip_ellipsis(phrase)
    already, remaining, typed_frac, fp, raw_orange = read_screen_progress(
        bgr,
        words=words,
        hop_phrases=hop_phrases,
        phrases=phrases,
        hint_phrase=phrase,
    )
    reset = typed_frac < 0.15 and bool(remaining) and not already
    return OcrResult(
        raw=raw_orange or "",
        matched=phrase,
        score=100.0,
        remaining=remaining
        if remaining
        else ("" if compact(already) == compact(phrase) else phrase),
        already_typed=already,
        parchment_present=True,
        progress_reset=reset,
        ink_fp=fp,
        typed_frac=typed_frac,
    )


def recognize(
    bgr: np.ndarray,
    phrases: list[str],
    min_score: float,
    lines: list[str] | None = None,
    words: list[str] | None = None,
    hop_phrases: list[str] | None = None,
) -> OcrResult:
    # Dialogue on black / non-parchment UI → do not OCR
    if not parchment_present(bgr):
        return OcrResult(
            raw="",
            matched="",
            score=0.0,
            remaining="",
            already_typed="",
            parchment_present=False,
        )

    black_m, red_m, all_m = split_typed_remaining_masks(bgr)
    black_m, red_m, all_m = restrict_masks_to_text_span(black_m, red_m, all_m)
    black_px = int((black_m > 0).sum())
    red_px = int((red_m > 0).sum())

    raw_all = _safe_tesseract_mask(all_m) or _safe_tesseract_bgr(bgr)
    raw_typed = _safe_tesseract_mask(black_m) if black_px > 40 else ""
    raw_remain = _safe_tesseract_mask(red_m) if red_px > 40 else ""

    # Line + hop composites (Witness The Cross Of The Lord Flee, …)
    phrase_bank = list(phrases)
    raw_c_all = compact(raw_all)
    if lines and any(k in raw_c_all for k in ("flee", "filee", "away", "hostil")):
        for line in lines:
            phrase_bank.append(f"{line} Flee")
            phrase_bank.append(f"{line} Flee Ye")
            phrase_bank.append(f"{line} Flee Ye Hostiles")
            phrase_bank.append(f"{line} Away")
            phrase_bank.append(f"{line} Away Ye Hostiles")

    matched, score = best_phrase_match(raw_all, phrase_bank, min_score, lines=lines)
    if not matched and (raw_typed or raw_remain):
        combined = f"{raw_typed} {raw_remain}".strip()
        matched, score = best_phrase_match(
            combined, phrase_bank, min_score, lines=lines
        )

    # Hop-table phrases — only win when they cover OCR better than a lang line.
    # Never replace a long line ("Witness The Cross Of The Lord") with a short
    # hop skeleton ("Witness Cross Flee Ye Hostiles").
    if words or hop_phrases:
        for raw_try in (raw_all, f"{raw_typed} {raw_remain}".strip(), raw_remain):
            if not raw_try:
                continue
            hop_hit, hop_score = match_hop_phrase(
                raw_try, hop_phrases or [], words or [], min_score
            )
            if not hop_hit or hop_score < min_score:
                continue
            hop_c = compact(hop_hit)
            matched_c = compact(matched) if matched else ""
            raw_c = compact(raw_try)
            core_c = ocr_core(raw_c)
            hop_covers = (
                _length_ok(core_c, hop_c)
                or _length_ok(raw_c, hop_c)
                or (len(hop_c) >= int(0.7 * len(core_c)) and len(hop_c) >= 10)
            )
            matched_fits = _length_ok(core_c, matched_c) if matched else False
            if not matched and hop_covers:
                matched, score = hop_hit, hop_score
            elif matched and hop_covers and (
                not matched_fits
                or (
                    abs(len(hop_c) - len(core_c)) < abs(len(matched_c) - len(core_c))
                    and hop_score + 5 >= score
                )
            ):
                matched, score = hop_hit, hop_score

    # Short solo hops: OCR literally is ForWeHaveHope — don't expand to Mercy line
    core_all = ocr_core(raw_c_all)
    if "forwehavehope" in core_all or core_all == "forwehavehope":
        matched, score = "For We Have Hope", max(score, 95.0)
    elif core_all in {"forwehope", "havehope"} or (
        "forwehope" in core_all
        and "mercy" not in core_all
        and "maythy" not in core_all
        and len(core_all) <= 14
    ):
        if "have" in core_all:
            matched, score = "For We Have Hope", max(score, 92.0)
        else:
            matched, score = "For We Hope", max(score, 90.0)

    # Witness line — OCR often ends Filee; force high confidence so bot locks
    core_n = (
        core_all.replace("filee", "flee")
        .replace("crossoff", "crossof")
    )
    if "witnessthecross" in core_n or (
        "witness" in core_n and "crossof" in core_n and "lord" in core_n
    ):
        if "flee" in core_n or "filee" in core_all:
            matched, score = "Witness The Cross Of The Lord Flee", max(score, 95.0)
        else:
            matched, score = "Witness The Cross Of The Lord", max(score, 92.0)

    # Glyph verification only when OCR match is weak (expensive)
    if score < 85:
        pack = load_primary_pack()
        if pack is not None:
            hints = [raw_all, f"{raw_typed} {raw_remain}".strip(), raw_remain, raw_typed]
            pool = _candidate_pool(
                phrases, lines, hop_phrases, words, hints, limit=30
            )
            if matched:
                pool = [matched] + [p for p in pool if p != matched]
            glyph_hit, glyph_score, _scale = best_phrase_by_glyphs(
                all_m,
                pool or ([matched] if matched else phrases[:30]),
                pack=pack,
                top_n=20,
            )
            if glyph_hit and (
                not matched
                or (
                    glyph_score >= max(42.0, score * 0.85)
                    and len(compact(glyph_hit)) >= len(compact(matched)) - 2
                )
            ):
                matched = glyph_hit
                score = max(score, glyph_score)

    # Lang line + hop word on same parchment (…Lord Flee / …Lord Away …)
    hop_suffixes = {"flee", "away", "hostiles", "ye"}
    if matched and words:
        for raw_src in (raw_all, raw_remain, f"{raw_typed} {raw_remain}".strip()):
            if not raw_src:
                continue
            leftover = compact_leftover(compact(raw_src), compact(matched))
            if not leftover:
                continue
            extra_word, extra_score = match_vocab_word(
                leftover, words, max(50.0, min_score - 15)
            )
            if (
                extra_word
                and extra_score >= 80
                and compact(extra_word) not in compact(matched)
                and (
                    compact(extra_word) in hop_suffixes
                    or extra_score >= 90
                    or leftover.endswith(compact(extra_word)[:4])
                )
            ):
                matched = f"{matched} {extra_word}".strip()
                score = max(score, extra_score)
                # Keep peeling hop leftovers (Flee Ye Hostiles)
                continue
            break

    # Only trust layout re-ranking when OCR text is usable (not chrome noise)
    raw_c_layout = compact(raw_all)
    ocr_ok = len(raw_c_layout) >= 10 and not looks_like_garbage(raw_all)

    layout_cands: list[str] = []
    if matched:
        layout_cands.append(matched)
        if not matched.lower().rstrip().endswith("flee"):
            layout_cands.append(f"{matched} Flee")

    core_layout = ocr_core(raw_c_layout)
    short_ocr = len(core_layout) <= 18

    if ocr_ok and lines and not short_ocr:
        scored_lines: list[tuple[float, str]] = []
        for line in lines:
            lc = compact(line)
            if len(lc) < 8:
                continue
            # Skip Latin liturgy unless OCR itself looks Latin-ish
            latinish = sum(1 for t in ("itaque", "populo", "sanctuario", "terribilis", "adesto", "cernunnos", "virtutem") if t in lc)
            if latinish and not any(t in raw_c_layout for t in ("itaque", "populo", "sanctuario", "terribilis", "adesto")):
                continue
            # Don't let partial_ratio promote a long prayer from a short OCR tail
            if len(lc) > len(core_layout) + max(8, len(core_layout) // 2):
                continue
            scored_lines.append((float(fuzz.partial_ratio(core_layout, lc)), line))
        scored_lines.sort(reverse=True)
        for sc, line in scored_lines[:5]:
            if sc < 70:
                break
            layout_cands.append(line)
            if not line.lower().rstrip().endswith("flee"):
                layout_cands.append(f"{line} Flee")

    # High-value composites only with strong OCR anchors (don't expand Hope → full Mercy line)
    if ocr_ok:
        if "witness" in raw_c_layout or "crossof" in raw_c_layout:
            layout_cands.append("Witness The Cross Of The Lord Flee")
            layout_cands.append("Witness The Cross Of The Lord")
        if "mercy" in raw_c_layout or "maythy" in raw_c_layout:
            layout_cands.append("May Thy Mercy O Lord Rest Upon Us For We Hope")
            layout_cands.append("May Thy Mercy O Lord Rest Upon Us For We Hope Flee")
        # Short hop visible alone on parchment
        if "forwehavehope" in core_layout or (
            "havehope" in core_layout and "mercy" not in core_layout
        ):
            layout_cands = [
                "For We Have Hope",
                "For We Hope",
                "Have Hope",
            ] + [c for c in layout_cands if compact(c) in {
                "forwehavehope", "forwehope", "havehope"
            } or len(compact(c)) <= len(core_layout) + 4]

    # Layout pick only improves *which phrase* — progress always comes from screen ink
    if ocr_ok and layout_cands:
        picked = _layout_pick_phrase(bgr, layout_cands, raw_hint=raw_all)
        if picked is not None:
            cand, _prog = picked
            cand_c = compact(cand)
            matched_c = compact(matched) if matched else ""
            cand_too_long = len(cand_c) > len(core_layout) + max(6, len(core_layout) // 2)
            if matched and (
                cand_too_long
                or len(cand_c) > len(matched_c) + max(6, len(matched_c) // 2)
            ):
                pass  # keep matched
            elif matched and score >= 88:
                overlap = float(fuzz.partial_ratio(matched_c, cand_c))
                if overlap >= 75:
                    matched, score = cand, max(score, 85.0)
            elif not cand_too_long:
                matched, score = cand, max(score, 80.0)

    already, remaining, typed_frac, fp, raw_orange = read_screen_progress(
        bgr,
        words=words,
        hop_phrases=hop_phrases,
        phrases=phrases,
        hint_phrase=matched or "",
    )

    # No dictionary hit — type what orange ink says
    if not matched and remaining:
        matched, score = remaining, max(score, 82.0)
    if not matched and not remaining:
        return OcrResult(
            raw=raw_all,
            matched="",
            score=score,
            remaining="",
            already_typed="",
            parchment_present=True,
            ink_fp=fp,
            typed_frac=typed_frac,
        )

    reset = typed_frac < 0.15 and bool(remaining) and not already
    if not remaining and typed_frac > 0.7:
        already = matched
        remaining = ""

    return OcrResult(
        raw=raw_all or raw_orange,
        matched=matched,
        score=score,
        remaining=(
            remaining
            if remaining
            else ("" if compact(already) == compact(matched) else matched)
        ),
        already_typed=already,
        parchment_present=True,
        progress_reset=reset,
        ink_fp=fp,
        typed_frac=typed_frac,
    )
