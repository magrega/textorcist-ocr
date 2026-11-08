"""Extract combat hop-words from Textorcist data.win + consecutive multi-word phrases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_DATA = Path(r"D:\SteamLibrary\steamapps\common\Textorcist\data.win")
ROOT = Path(__file__).resolve().parents[1]

SKIP_PREFIXES = ("stance", "machine", "classic", "fork", "arrow", "grenade", "missile", "bomb")


def _read_len_strings(chunk: bytes) -> list[str]:
    words: list[str] = []
    pos = 0
    while pos + 5 < len(chunk):
        ln = int.from_bytes(chunk[pos : pos + 4], "little")
        if 2 <= ln <= 28 and pos + 4 + ln <= len(chunk):
            raw = chunk[pos + 4 : pos + 4 + ln]
            if all((65 <= b <= 90) or (97 <= b <= 122) or b in (32, 39, 33) for b in raw):
                s = raw.decode("ascii")
                if (
                    s[0].isupper()
                    and "_" not in s
                    and " " not in s
                    and not any(s.lower().startswith(p) for p in SKIP_PREFIXES)
                ):
                    words.append(s)
                    pos = pos + 4 + ln
                    if pos < len(chunk) and chunk[pos] == 0:
                        pos += 1
                    continue
        pos += 1
    return words


def extract_words(data: bytes) -> tuple[list[str], list[str]]:
    # Primary hop table
    start = data.find(b"lastHopIndex")
    if start < 0:
        start = data.find(b"HopIndex")
    hop_seq = _read_len_strings(data[start : start + 100000]) if start >= 0 else []

    # Broader scan for TitleCase tokens (catches Away, etc. outside hop table)
    broad = _read_len_strings(data[5_000_000:8_000_000])

    seen: set[str] = set()
    words: list[str] = []
    for w in hop_seq + broad:
        if w not in seen:
            seen.add(w)
            words.append(w)

    # Consecutive hop n-grams (Flee Ye Hostiles, Ye Hostiles, ...)
    phrases: list[str] = []
    # Prefer the hop table order near Witness/Cross/Flee
    seq = hop_seq
    # Dedup consecutive while preserving order for n-grams
    clean_seq: list[str] = []
    for w in seq:
        if not clean_seq or clean_seq[-1] != w:
            # stop at obvious non-dialogue tokens
            if w.lower() in {"hop", "disappear", "reappear", "matthew"}:
                break
            if w.startswith("stance") or "_" in w:
                break
            clean_seq.append(w)

    for n in range(2, 6):
        for i in range(0, max(0, len(clean_seq) - n + 1)):
            phrases.append(" ".join(clean_seq[i : i + n]))

    # Also n-grams from a window around Away if present in broad
    if "Away" in seen:
        # Build small local sequences that include Away + common followers
        for extra in (
            "Away Ye Hostiles",
            "Away Ye",
            "Ye Hostiles",
            "Flee Ye Hostiles",
            "Flee Ye",
        ):
            phrases.append(extra)

    # unique phrases
    p_seen: set[str] = set()
    uniq_phrases: list[str] = []
    for p in phrases:
        if p not in p_seen:
            p_seen.add(p)
            uniq_phrases.append(p)

    return words, uniq_phrases


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    words, hop_phrases = extract_words(path.read_bytes())
    (ROOT / "words.json").write_text(json.dumps(words, indent=2), encoding="utf-8")
    (ROOT / "hop_phrases.json").write_text(
        json.dumps(hop_phrases, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(words)} words, {len(hop_phrases)} hop phrases")
    print("Away" in words, "Flee" in words, "Hostiles" in words)
    print([p for p in hop_phrases if "Hostiles" in p or "Away" in p][:20])


if __name__ == "__main__":
    main()
