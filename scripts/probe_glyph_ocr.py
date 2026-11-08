"""Probe connected-component glyph classification on debug_roi.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr_engine import split_typed_remaining_masks  # noqa: E402

PACK = ROOT / "glyphs" / "sprite_fonts" / "fontAlagard"


def main() -> None:
    im = cv2.imread(str(ROOT / "debug_roi.png"))[:, :1000]
    _t, rem, _a = split_typed_remaining_masks(im)
    h, w = rem.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1))
    clean = cv2.subtract(rem, cv2.morphologyEx(rem, cv2.MORPH_OPEN, hk))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    row = (clean > 0).sum(1)
    peak = int(row.argmax())
    y1, y2 = max(0, peak - 45), min(h, peak + 45)
    band_full = clean[y1:y2]
    col = (band_full > 0).sum(0)
    xs = np.where(col > 2)[0]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    band = band_full[:, x1:x2]
    cv2.imwrite(str(ROOT / "debug_probe_band.png"), band)

    n, labels, stats, _cent = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        if area < 40 or bw < 4 or bh < 10:
            continue
        if bw > band.shape[1] * 0.5:
            continue
        comps.append((x, y, bw, bh, area, i))
    comps.sort(key=lambda c: c[0])
    print("components", len(comps), [(c[0], c[2], c[3], c[4]) for c in comps[:20]])

    meta = json.loads((PACK / "metrics.json").read_text(encoding="utf-8"))
    templates: dict[str, np.ndarray] = {}
    for g in meta["glyphs"]:
        ch = g["char"]
        if not ch.isascii() or ch in " .":
            continue
        raw = cv2.imread(str(PACK / g["file"]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        templates[ch] = (raw[:, :, 3] > 0).astype(np.uint8) * 255

    out = []
    prev_x2 = None
    for x, y, bw, bh, area, idx in comps:
        blob = (labels[y : y + bh, x : x + bw] == idx).astype(np.uint8) * 255
        # trim
        ys, xs2 = np.where(blob > 0)
        blob = blob[ys.min() : ys.max() + 1, xs2.min() : xs2.max() + 1]
        bh2, bw2 = blob.shape
        best = None
        for ch, t in templates.items():
            # resize template to blob height, keep aspect
            scale = bh2 / t.shape[0]
            tw = max(2, int(round(t.shape[1] * scale)))
            th = bh2
            rt = cv2.resize(t, (tw, th), interpolation=cv2.INTER_NEAREST)
            # pad both to same size
            H = max(th, bh2)
            W = max(tw, bw2)
            a = np.zeros((H, W), np.uint8)
            b = np.zeros((H, W), np.uint8)
            a[:bh2, :bw2] = blob
            b[:th, :tw] = rt
            # also try a few x offsets
            for dx in range(-2, 3):
                b2 = np.zeros_like(a)
                x0 = max(0, dx)
                x1b = min(W, tw + dx)
                if x1b <= x0:
                    continue
                src = rt[:, : x1b - x0] if dx >= 0 else rt[:, x0 - dx : tw]
                # simpler: matchTemplate if sizes allow
            inter = np.logical_and(a > 0, b > 0).sum()
            union = np.logical_or(a > 0, b > 0).sum()
            iou = inter / union if union else 0
            # correlation on same canvas
            anum = a.astype(np.float32)
            bnum = b.astype(np.float32)
            anum -= anum.mean()
            bnum -= bnum.mean()
            denom = (np.linalg.norm(anum) * np.linalg.norm(bnum)) + 1e-6
            corr = float((anum * bnum).sum() / denom)
            score = 0.5 * iou + 0.5 * max(0, corr)
            if best is None or score > best[0]:
                best = (score, ch, iou, corr)
        if prev_x2 is not None and x - prev_x2 > max(12, bw * 0.6):
            out.append(" ")
        ch = best[1] if best and best[0] >= 0.25 else "?"
        out.append(ch)
        print("comp", x, bw, bh, "->", best)
        prev_x2 = x + bw
    print("OCR:", repr("".join(out)))


if __name__ == "__main__":
    main()
