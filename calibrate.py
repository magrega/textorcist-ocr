"""
Calibrate parchment ROI without needing the game to stay focused.

Default flow (recommended):
  1. Start this script
  2. Click into Textorcist so it is focused / visible
  3. Press F6 — captures the game window instantly
  4. Select the parchment box on the frozen snapshot (game may minimize; that's fine)

Also works from a Steam screenshot:
  py -3.10 calibrate.py path\\to\\shot.png
"""
from __future__ import annotations

# DPI awareness before any screen coordinate / GUI work (4K + scaling)
from window_capture import enable_dpi_awareness, find_game_window, grab_game_frame

enable_dpi_awareness()

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def wait_for_f6_capture(process_name: str, title_substr: str) -> np.ndarray:
    """Wait until user presses F6 while the game is up, then grab that window."""
    result: dict = {"frame": None, "error": None}
    ready = threading.Event()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        if key != keyboard.Key.f6:
            return
        try:
            # Tiny delay so the keypress itself isn't an issue
            time.sleep(0.05)
            frame, win = grab_game_frame(
                process_name=process_name,
                title_substr=title_substr,
                focus=False,
            )
            # Sanity: reject near-black frames (minimized / covered)
            if float(np.mean(frame)) < 5.0:
                result["error"] = (
                    "Capture was almost black — game was probably minimized. "
                    "Focus the game, make sure you can see it, then press F6 again."
                )
                return
            result["frame"] = frame
            print(
                f"Captured '{win.title}' "
                f"region=({win.left},{win.top}) {win.width}x{win.height} "
                f"image={frame.shape[1]}x{frame.shape[0]}"
            )
            ready.set()
            return False  # stop listener
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            # keep listening so user can retry after launching game

    print(f"Target: {process_name}")
    print("1) Focus the game so it is visible on screen")
    print("2) Press F6 to snapshot the game window")
    print("3) Then draw the parchment ROI on the frozen image")
    print("   (Esc in this console with Ctrl+C to abort)")

    # Verify game exists at least once up front (non-fatal if it appears later)
    try:
        win = find_game_window(process_name, title_substr)
        print(f"Found window '{win.title}' — switch to it, then press F6.")
    except RuntimeError as exc:
        print(f"[wait] {exc}")

    with keyboard.Listener(on_press=on_press) as listener:
        while not ready.is_set():
            if result["error"]:
                print(f"[retry] {result['error']}")
                result["error"] = None
            time.sleep(0.1)
        listener.join(timeout=1.0)

    frame = result["frame"]
    if frame is None:
        raise SystemExit("Capture failed.")
    return frame


def select_and_save(frame: np.ndarray, cfg: dict) -> None:
    snap_path = ROOT / "debug_capture.png"
    cv2.imwrite(str(snap_path), frame)
    print(f"Snapshot saved -> {snap_path}")

    win_name = "Select parchment ROI (ENTER=save, ESC=cancel)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, frame)
    roi = cv2.selectROI(win_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    if w < 10 or h < 10:
        raise SystemExit("ROI too small / cancelled.")

    fh, fw = frame.shape[:2]
    cfg["roi"] = {
        "rel_left": round(x / fw, 4),
        "rel_top": round(y / fh, 4),
        "rel_right": round((x + w) / fw, 4),
        "rel_bottom": round((y + h) / fh, 4),
    }
    save_config(cfg)
    print("Saved ROI (relative to game window):", cfg["roi"])
    crop = frame[y : y + h, x : x + w]
    out = ROOT / "debug_roi.png"
    cv2.imwrite(str(out), crop)
    print(f"Wrote preview -> {out}")


def main() -> None:
    cfg = load_config()
    process_name = cfg.get("process_name", "Textorcist.exe")
    title_substr = cfg.get("window_title_substr", "Textorcist")

    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            raise SystemExit(f"Could not read image: {sys.argv[1]}")
        print(f"Using screenshot: {sys.argv[1]}")
    else:
        frame = wait_for_f6_capture(process_name, title_substr)

    select_and_save(frame, cfg)


if __name__ == "__main__":
    main()
