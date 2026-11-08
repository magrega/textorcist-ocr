"""Extract typeable combat lines from Textorcist lang.json into lines.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_LANG = Path(r"D:\SteamLibrary\steamapps\common\Textorcist\lang.json")
ROOT = Path(__file__).resolve().parents[1]


def load_lang(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    cleaned = raw.replace('\\"', '"')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'"En"\s*:\s*\{(.*)\}\s*,?\s*"', cleaned, re.S)
        if not m:
            raise
        return {"En": json.loads("{" + m.group(1) + "}")}


def word_chunks(text: str) -> set[str]:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    out: set[str] = set()
    if not words:
        return out
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            if j - i == 1 and len(words[i]) <= 1 and len(words) > 1:
                # keep lone "I" — it appears as its own parchment beat
                out.add(words[i])
                continue
            out.add(" ".join(words[i:j]))
    return out


def extract(lang: dict) -> tuple[list[str], list[str]]:
    en = lang.get("En", lang)
    phrases: set[str] = set()
    lines: set[str] = set()

    for key, value in en.items():
        if not isinstance(value, str):
            continue
        if re.fullmatch(r"E\d+L\d+", key):
            text = value.strip()
            lines.add(text)
            phrases |= word_chunks(text)
        for seg in re.findall(r"\[S\](.*?)\[S\]", value, flags=re.I):
            cleaned = re.sub(r"\s+", " ", seg).strip()
            if cleaned:
                lines.add(cleaned)
                phrases |= word_chunks(cleaned)
        if key.endswith("WRD"):
            phrases.add(value.strip())

    return (
        sorted(phrases, key=lambda s: (len(s), s.lower())),
        sorted(lines, key=lambda s: (len(s), s.lower())),
    )


def main() -> None:
    lang_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LANG
    _, lines = extract(load_lang(lang_path))
    (ROOT / "lines.json").write_text(
        json.dumps(lines, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(lines)} full lines")


if __name__ == "__main__":
    main()
