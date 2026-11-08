"""
Textorcist auto-typer.

Reads the battle parchment directly from screen pixels (see parchment_reader):
every letter is read against the game's own font sprites and classified as
typed (navy + white outline) or untyped (brown). The bot types exactly what is
still brown, so it resumes correctly after book drops / interruptions and
notices immediately when a new line appears.

Hotkeys:
  F8  toggle typing on/off
  F7  save a debug snapshot of the parchment
  F9  quit
"""
from __future__ import annotations

from window_capture import enable_dpi_awareness, find_game_window

enable_dpi_awareness()

import json
import os
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput.keyboard import Controller, GlobalHotKeys, Key

from glyph_matcher import load_primary_pack
from parchment_reader import (
    ParchmentLine,
    build_vocab,
    correct_with_vocab,
    debug_overlay,
    read_parchment,
)

ROOT = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "TextorcistOCRHelper"
    ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {
    "auto_roi": True,
    "process_name": "Textorcist.exe",
    "window_title_substr": "Textorcist",
    "roi": {"rel_left": 0.149, "rel_top": 0.9065, "rel_right": 0.8505, "rel_bottom": 1.0},
    "poll_ms": 50,
    "key_delay_ms": 12,
    "stall_ms": 300,
    "stable_frames": 2,
}
VOCAB_FILES = ("words.json", "hop_phrases.json", "lines.json")


def _load_vocab() -> dict[int, dict[str, str]]:
    words: list[str] = []
    for name in VOCAB_FILES:
        path = ROOT / name
        if path.exists():
            for entry in json.loads(path.read_text(encoding="utf-8")):
                words.extend(str(entry).split())
    return build_vocab(words)


class Bot:
    def __init__(self) -> None:
        if CONFIG_PATH.exists():
            self.cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            self.cfg = DEFAULT_CONFIG.copy()
            CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")
        self.vocab = _load_vocab()
        self.keyboard = Controller()
        self.enabled = False
        self.running = True
        self.debug_once = False
        self.key_delay = float(self.cfg.get("key_delay_ms", 12)) / 1000.0
        # Time the screen may show no progress after a burst before we resend
        self.stall_s = float(self.cfg.get("stall_ms", 300)) / 1000.0
        # Frames the same new text must be seen before typing it (anti-flicker)
        self.stable_frames = int(self.cfg.get("stable_frames", 2))

        self._status = ""
        self._paused = True
        self._line_text = ""
        self._candidate_text = ""
        self._candidate_seen = 0
        self._last_typed = -1
        self._last_change = 0.0
        self._sent_upto = -1
        self._stall_index = -1
        self._stall_count = 0
        self._idle_shown = ""

    # ── hotkeys ──────────────────────────────────────────────────────────
    def toggle(self) -> None:
        self.enabled = not self.enabled
        self._reset_send_state()
        self._idle_shown = ""
        print(f"[bot] typing {'ON' if self.enabled else 'OFF'}")

    def quit(self) -> None:
        self.running = False
        print("[bot] quitting...")

    def dump(self) -> None:
        self.debug_once = True

    # ── helpers ──────────────────────────────────────────────────────────
    def set_status(self, msg: str) -> None:
        if msg != self._status:
            self._status = msg
            print(msg)

    def _reset_send_state(self) -> None:
        self._sent_upto = -1
        self._stall_index = -1
        self._stall_count = 0

    def type_text(self, text: str) -> None:
        for ch in text:
            if not self.running or not self.enabled:
                return
            # Focus can change halfway through a burst (e.g. opening the helper).
            try:
                win = find_game_window(
                    self.cfg.get("process_name", "Textorcist.exe"),
                    self.cfg.get("window_title_substr", "Textorcist"),
                )
                if win.minimized or not win.foreground:
                    return
            except RuntimeError:
                return
            if ch == " ":
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
            else:
                self.keyboard.type(ch)
            time.sleep(self.key_delay)

    def save_debug(self, roi: np.ndarray, line: ParchmentLine | None, tag: str) -> None:
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(ROOT / f"debug_{tag}_{stamp}.png"), roi)
        cv2.imwrite(str(ROOT / f"debug_{tag}_{stamp}_boxes.png"), debug_overlay(roi, line))
        desc = line.describe() if line else "no parchment text"
        print(f"[debug] saved debug_{tag}_{stamp}.png  (UPPER=typed, lower=untyped: {desc})")

    # ── main per-frame logic ─────────────────────────────────────────────
    def process_frame(self, roi: np.ndarray) -> None:
        now = time.time()
        line = read_parchment(roi)
        if line is not None:
            correct_with_vocab(line, self.vocab)

        if self.debug_once:
            self.debug_once = False
            self.save_debug(roi, line, "f7")

        if line is None:
            if not self._paused:
                print("[paused] no parchment text (book dropped / dialogue / between lines)")
            self._paused = True
            self._reset_send_state()
            return

        text = line.text
        typed = line.typed_count
        total = line.total

        if self._paused:
            self._paused = False
            print(f"[resume] parchment back: {line.describe()!r} ({typed}/{total} typed)")
            self._reset_send_state()

        if text == self._line_text:
            self._candidate_text, self._candidate_seen = "", 0
            if typed < self._last_typed:
                # Same text but progress went back: line repeated or reset by the game
                print(f"[line] {text!r} restarted ({typed}/{total} typed)")
                self._reset_send_state()
                self._last_typed = typed
                self._last_change = now
        else:
            # Text changed: a new line. Require it on consecutive frames (anti-flicker)
            if text != self._candidate_text:
                self._candidate_text, self._candidate_seen = text, 0
            self._candidate_seen += 1
            if self._candidate_seen < self.stable_frames:
                return
            print(f"[line] {text!r} ({typed}/{total} typed)")
            self._line_text = text
            self._candidate_text, self._candidate_seen = "", 0
            self._reset_send_state()
            self._last_typed = typed
            self._last_change = now
            self._idle_shown = ""

        if typed != self._last_typed:
            self._last_typed = typed
            self._last_change = now

        if typed >= total:
            return

        remain = line.remaining(typed)
        if not self.enabled:
            if remain != self._idle_shown:
                self._idle_shown = remain
                print(f"[idle] next to type: {remain!r} ({typed}/{total})  — F8 to start")
            return

        # Waiting for the screen to catch up with what we already sent
        if self._sent_upto > typed and now - self._last_change < self.stall_s:
            return

        prefix = ""
        if self._sent_upto > typed:
            # Screen did not advance past `typed` — the game rejected that key
            if typed == self._stall_index:
                self._stall_count += 1
            else:
                self._stall_index, self._stall_count = typed, 1
            remain, prefix = self._stall_fix(line, typed, remain)
            if self._stall_count == 3:
                self.save_debug(roi, line, "stuck")

        send = prefix + remain
        print(f"[type] {send!r}  ({typed}/{total} typed)")
        self.type_text(send)
        self._sent_upto = total
        self._last_change = time.time()

    def _stall_fix(self, line: ParchmentLine, typed: int, remain: str) -> tuple[str, str]:
        """Pick a different strategy for each consecutive stall at the same letter."""
        n = self._stall_count
        glyph = line.letters[typed]
        print(f"[stall] screen stuck at letter {typed} ({glyph.char!r}), attempt {n}")
        if n == 2 and line.at_word_start(typed):
            return remain, " "
        alts = glyph.alts
        if n >= 3 and alts:
            alt = alts[(n - 3) % len(alts)]
            print(f"[stall] trying {alt!r} instead of {glyph.char!r}")
            return alt + remain[1:], ""
        return remain, ""

    # ── capture loop ─────────────────────────────────────────────────────
    def roi_region(self, win) -> dict:
        r = self.cfg["roi"]
        x1 = int(r["rel_left"] * win.width)
        y1 = int(r["rel_top"] * win.height)
        x2 = int(r["rel_right"] * win.width)
        y2 = int(r["rel_bottom"] * win.height)
        return {
            "left": win.left + x1,
            "top": win.top + y1,
            "width": max(1, x2 - x1),
            "height": max(1, y2 - y1),
        }

    def run(self) -> None:
        process_name = self.cfg.get("process_name", "Textorcist.exe")
        title_substr = self.cfg.get("window_title_substr", "Textorcist")
        pack = load_primary_pack()
        print("Textorcist auto-typer")
        print(f"  target: {process_name}")
        print(f"  font sprites: {pack.name if pack else 'MISSING (run scripts/build_glyph_pack.py)'}")
        print("  F8 = typing on/off | F7 = save debug snapshot | F9 = quit")
        print("  Reads letters + typed/untyped colour straight from the parchment.")
        print("  *** Leave the GAME focused. Do not click this console. ***")
        print()
        if pack is None:
            return

        hotkeys = GlobalHotKeys({"<f8>": self.toggle, "<f9>": self.quit, "<f7>": self.dump})
        hotkeys.start()
        poll = float(self.cfg.get("poll_ms", 50)) / 1000.0
        try:
            with mss.MSS() as sct:
                while self.running:
                    t0 = time.time()
                    try:
                        win = find_game_window(process_name, title_substr)
                    except RuntimeError as exc:
                        self.set_status(f"[wait] {exc}")
                        time.sleep(0.5)
                        continue
                    if win.minimized or not win.foreground:
                        self.set_status("[paused] Game not focused/minimized — click into Textorcist")
                        time.sleep(0.25)
                        continue
                    self.set_status(f"[live] '{win.title}' {win.width}x{win.height}")
                    try:
                        shot = np.array(sct.grab(self.roi_region(win)))
                        roi = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                        self.process_frame(roi)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[error] {exc}")
                    time.sleep(max(0.0, poll - (time.time() - t0)))
        except KeyboardInterrupt:
            print("\n[bot] stopped (Ctrl+C)")
        finally:
            self.running = False
            try:
                hotkeys.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    Bot().run()
