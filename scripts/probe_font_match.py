"""Find which glyph pack matches the stuck Witness screenshot best."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "debug_witness_stuck.png"


def load_templates(folder: Path) -> dict[str, np.ndarray]:
    meta = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    tmps: dict[str, np.ndarray] = {}
    for g in meta["glyphs"]:
        ch = g["char"]
        if not isinstance(ch, str) or len(ch) != 1 or not ch.isascii() or ch == " ":
            continue
        path = folder / g.get("file", f"U+{ord(ch):04X}.png")
        if not path.exists():
            path = folder / f"U+{ord(ch):04X}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            m = (raw[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            m = (raw > 40).astype(np.uint8) * 255
        if int(m.sum()) > 0:
            tmps[ch] = m
    return tmps


def main() -> None:
    im = cv2.imread(str(IMG))
    B, G, R = cv2.split(im)
    parch = (
        (R > 160)
        & (G > 150)
        & (B > 120)
        & (R >= G - 15)
        & (G >= B - 20)
        & ((R.astype(np.int16) - B.astype(np.int16)) > 10)
    )
    row = parch.mean(axis=1)
    ys = np.where(row > 0.4)[0]
    roi = im[int(ys.min()) : int(ys.max()) + 1]
    B, G, R = cv2.split(roi)
    remain = (
        (R > 150) & (R > G + 50) & (R > B + 50) & (G < 140) & (B < 140)
    ).astype(np.uint8) * 255
    remain = cv2.morphologyEx(remain, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    row = remain.sum(axis=1) // 255
    peak = int(np.argmax(row))
    y1, y2 = max(0, peak - 30), min(remain.shape[0], peak + 30)
    band = remain[y1:y2]
    xs = np.where(band.sum(axis=0) > 0)[0]
    band = band[:, int(xs.min()) : int(xs.max()) + 1]
    print("band", band.shape)
    cv2.imwrite(str(ROOT / "debug_witness_band.png"), band)

    packs = [
        "glyphs/sprite_fonts/fontAlagard",
        "glyphs/sprite_fonts/fontAlagardOutline",
        "glyphs/sprite_fonts/fontCeltic2x",
        "glyphs/sprite_fonts/fontCeltic",
        "glyphs/sprite_fonts/fontGibberesque",
        "glyphs/sprite_fonts/fontRomulus",
        "glyphs/fonts/fnt_alagard_16",
        "glyphs/fonts/fnt_celtic_8",
        "glyphs/fonts/fnt_romulus_16",
        "glyphs/fonts/fnt_gibberesque_16",
    ]
    for rel in packs:
        folder = ROOT / rel
        if not (folder / "metrics.json").exists():
            print(rel, "MISSING")
            continue
        tmps = load_templates(folder)
        hits = []
        for ch in "TCheosWrLfd":
            if ch not in tmps:
                continue
            tmpl = tmps[ch]
            local = []
            for scale in np.linspace(1.2, 10.0, 45):
                tw = max(2, int(tmpl.shape[1] * scale))
                th = max(2, int(tmpl.shape[0] * scale))
                if th >= band.shape[0] or tw >= band.shape[1]:
                    continue
                rt = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_NEAREST)
                v = float(cv2.matchTemplate(band, rt, cv2.TM_CCOEFF_NORMED).max())
                local.append((v, scale))
            if local:
                local.sort(reverse=True)
                hits.append((local[0][0], ch, local[0][1]))
        hits.sort(reverse=True)
        avg = float(np.mean([h[0] for h in hits[:5]])) if hits else 0
        print(f"{rel}: avg5={avg:.3f} top={hits[:5]}")


if __name__ == "__main__":
    main()
