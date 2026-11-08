"""Offline simulation of the bot against synthetic parchment frames.

Renders lines with the game's font sprites (brown = untyped, navy + white
outline = typed), models the game advancing one letter per correct key, and
checks that the bot finishes lines, resumes after a book drop, and notices
line changes. Run: py -3.10 scripts/sim_bot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bot as bot_mod  # noqa: E402
from glyph_matcher import load_primary_pack  # noqa: E402

UNIT = 6
PARCH = (184, 200, 216)
GUIDE = (136, 200, 216)
BROWN = (40, 88, 152)
NAVY = (72, 40, 40)
WHITE = (248, 248, 248)
DESCENDERS = set("gjpqy")


def render(text: str, typed: int, w: int = 2694, h: int = 202) -> np.ndarray:
    pack = load_primary_pack()
    img = np.full((h, w, 3), PARCH, np.uint8)
    for y in range(40, h, 30):
        img[y : y + 6, 60 : w - 60] = GUIDE
    baseline = 142
    x = 124
    letter_i = 0
    typed_mask = np.zeros((h, w), bool)
    brown_mask = np.zeros((h, w), bool)
    for ch in text:
        if ch == " ":
            x += 13 * UNIT
            continue
        t = pack.templates[ch] > 0
        ys, xs = np.nonzero(t)
        t = t[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        big = np.kron(t, np.ones((UNIT, UNIT), bool))
        gh, gw = big.shape
        bottom = baseline + (3 * UNIT if ch in DESCENDERS else 0)
        y0 = bottom - gh
        region = np.zeros((h, w), bool)
        region[y0 : y0 + gh, x : x + gw] = big
        if ch != "." and letter_i < typed:
            typed_mask |= region
        else:
            brown_mask |= region
        if ch != ".":
            letter_i += 1
        x += gw + 2 * UNIT
    outline = cv2.dilate(typed_mask.astype(np.uint8), np.ones((2 * UNIT + 1,) * 2, np.uint8)) > 0
    img[outline & ~typed_mask] = WHITE
    img[brown_mask] = BROWN
    img[typed_mask] = NAVY
    return img


class FakeGame:
    def __init__(self, text: str, drop_key_at: int | None = None) -> None:
        self.text = text
        self.letters = [c for c in text if c not in " ."]
        self.pos = 0
        self.drop_key_at = drop_key_at
        self.keys = ""

    def key(self, ch: str) -> None:
        self.keys += ch
        if ch == " " or self.pos >= len(self.letters):
            return
        if self.drop_key_at is not None and self.pos == self.drop_key_at:
            self.drop_key_at = None  # simulate one lost key press
            return
        if ch == self.letters[self.pos]:
            self.pos += 1

    def frame(self) -> np.ndarray:
        return render(self.text, self.pos)


def run_frames(b: bot_mod.Bot, game: FakeGame, frames: int, clock: list[float], blank: bool = False) -> None:
    for _ in range(frames):
        clock[0] += 0.05
        roi = np.zeros((202, 2694, 3), np.uint8) if blank else game.frame()
        b.process_frame(roi)


def main() -> None:
    clock = [1000.0]
    bot_mod.time.time = lambda: clock[0]  # deterministic clock
    bot_mod.time.sleep = lambda s: None
    b = bot_mod.Bot()
    b.enabled = True
    game = FakeGame("...Away Ye Hostiles")

    def fake_type(text: str) -> None:
        for ch in text:
            game.key(ch)
            clock[0] += b.key_delay

    b.type_text = fake_type
    results = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}: {name}  (game pos {game.pos}/{len(game.letters)}, keys sent {game.keys!r})")

    # 1) Fresh line with leading '...': nothing pre-typed, finishes the line
    run_frames(b, game, 6, clock)
    check("fresh '...Away Ye Hostiles' typed fully, starting at A", game.pos == len(game.letters) and game.keys.startswith("Away"))

    # 2) Book dropped mid-line, then picked up: resumes at the right letter
    game.__init__("Witness The Cross Of The Lord Flee...")
    game.pos = 22  # "Witness The Cross Of The" + "Lo" already navy
    keys_before = len(game.keys)
    run_frames(b, game, 3, clock, blank=True)
    run_frames(b, game, 6, clock)
    sent = game.keys[keys_before:]
    check("resume after book drop continues with 'rd Flee'", game.pos == len(game.letters) and sent.startswith("rd Flee"))

    # 3) Line changes without a gap: bot notices and types the new line
    game.__init__("Shut Up")
    run_frames(b, game, 6, clock)
    check("new line 'Shut Up' detected and typed", game.pos == len(game.letters))

    # 4) Same text appears again (repeated hop word): progress drop is a new line
    game.__init__("Shut Up")
    run_frames(b, game, 6, clock)
    check("repeated identical line typed again", game.pos == len(game.letters))

    # 5) A key press is lost by the game: bot notices the stall and resends
    game.__init__("For We Have Hope", drop_key_at=5)
    run_frames(b, game, 20, clock)
    check("recovers from a dropped key press", game.pos == len(game.letters))

    print()
    print(f"{sum(ok for _n, ok in results)}/{len(results)} scenarios passed")


if __name__ == "__main__":
    main()
