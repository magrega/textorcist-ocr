"""OCR smoke test against a screenshot (no typing)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pytesseract

from ocr_engine import preprocess, recognize, run_tesseract

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    phrases = json.loads((ROOT / "phrases.json").read_text(encoding="utf-8"))
    tess = cfg.get("tesseract_cmd") or ""
    if tess:
        pytesseract.pytesseract.tesseract_cmd = tess

    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if img_path is None:
        # Prefer calibrated crop, else look for any png in assets
        candidates = [
            ROOT / "debug_roi.png",
            *ROOT.glob("assets/**/*.png"),
            *ROOT.glob("*.png"),
        ]
        img_path = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        raise SystemExit("Pass a screenshot path: py -3.10 smoke_test.py shot.png")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"Could not read {img_path}")

    # If full screenshot, crop with relative ROI
    if frame.shape[0] > 400:
        r = cfg["roi"]
        h, w = frame.shape[:2]
        x1, y1 = int(r["rel_left"] * w), int(r["rel_top"] * h)
        x2, y2 = int(r["rel_right"] * w), int(r["rel_bottom"] * h)
        crop = frame[y1:y2, x1:x2]
    else:
        crop = frame

    cv2.imwrite(str(ROOT / "debug_smoke_crop.png"), crop)
    cv2.imwrite(str(ROOT / "debug_smoke_pre.png"), preprocess(crop))

    lines = json.loads((ROOT / "lines.json").read_text(encoding="utf-8")) if (ROOT / "lines.json").exists() else []
    words = json.loads((ROOT / "words.json").read_text(encoding="utf-8")) if (ROOT / "words.json").exists() else []
    hops = json.loads((ROOT / "hop_phrases.json").read_text(encoding="utf-8")) if (ROOT / "hop_phrases.json").exists() else []

    raw = run_tesseract(crop)
    result = recognize(
        crop,
        phrases,
        float(cfg.get("fuzzy_min_score", 70)),
        lines=lines,
        words=words,
        hop_phrases=hops,
    )
    print(f"image: {img_path}")
    print(f"parchment: {result.parchment_present}")
    print(f"raw:   {raw!r}")
    print(f"match: {result.matched!r} ({result.score:.1f})")
    print(f"already: {result.already_typed!r}")
    print(f"remain:  {result.remaining!r}")


if __name__ == "__main__":
    main()
