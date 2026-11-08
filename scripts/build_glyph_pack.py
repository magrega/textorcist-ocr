"""Build a curated glyph pack from extracted Textorcist assets.

Reads gm-extract output under extracted/ and writes glyphs/ with:
  - TrueType/bitmap FONT atlases + per-glyph PNGs + metrics.json
  - Sprite fonts remapped via scr_loadSpriteFonts charset strings
  - charset + font registry metadata for template OCR
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from gm_extract.gamedata import GameData
from gm_extract.resources import parse_fonts
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "extracted"
OUT = ROOT / "glyphs"
DATA_WIN = Path(r"D:\SteamLibrary\steamapps\common\Textorcist\data.win")

# Variable name -> (sprite, map line index 0-based in strings.txt)
# Ordering inferred from scr_loadSpriteFonts string table adjacency.
SPRITE_FONT_MAP = {
    "fontCeltic": ("spr_font_celtic", 856),
    "fontCeltic2x": ("spr_font_celtic_2x", 856),
    "fontCelticOutline": ("spr_font_celtic_outline", 863),
    "fontCelticOutline2x": ("spr_font_celtic_outline_2x", 863),
    "fontAlagard": ("spr_font_alagard", 866),
    "fontAlagardOutline": ("spr_font_alagard_outline", 868),
    "fontGibberesque": ("spr_font_gibberesque", 870),
    "fontGibberesqueOutline": ("spr_font_gibberesque_outline", 872),
    "fontGibberesque2x": ("spr_font_gibberesque_2x", 872),
    "fontGibberesqueOutline2x": ("spr_font_gibberesque_outline_2x", 872),
    "fontRomulus": ("spr_font_romulus", 876),
    "fontHeinz": ("spr_font_heinz", 878),  # "0123456789.,:;"
    "fontEasy": ("spr_font_easyread", 880),
    "fontEasyOutline": ("spr_font_easyread_outline", 880),
    "fontEasy2x": ("spr_font_easyread_2x", 880),
    "fontEasyOutline2x": ("spr_font_easyread_outline_2x", 880),
    "fontCelticDemon": ("spr_font_celtic_demon", 885),
    "fontCelticDemonOutline": ("spr_font_celtic_demon_outline", 887),
    "fontCelticDemonOutline2x": ("spr_font_celtic_demon_outline_2x", 887),
    "fontMap": ("spr_font_worldmap", 890),
}

# Combat / phrase UI priorities (Normal vs Demon)
PRIMARY_SPRITE_FONTS = [
    "fontCeltic",
    "fontCeltic2x",
    "fontCelticOutline",
    "fontCelticOutline2x",
    "fontCelticDemon",
    "fontCelticDemonOutline",
    "fontCelticDemonOutline2x",
    "fontAlagard",
    "fontAlagardOutline",
    "fontGibberesque",
    "fontGibberesque2x",
    "fontRomulus",
    "fontEasy",
    "fontEasy2x",
]


def load_strings() -> list[str]:
    return (EXTRACTED / "strings.txt").read_text(encoding="utf-8").splitlines()


def safe_filename(ch: str) -> str:
    code = ord(ch)
    if ch == " ":
        return "U+0020_space"
    if ch in '<>:"/\\|?*':
        return f"U+{code:04X}"
    if ch.isprintable() and ch not in ". ":
        return f"U+{code:04X}_{ch}"
    return f"U+{code:04X}"


def export_font_metrics() -> list[dict]:
    g = GameData(str(DATA_WIN))
    fonts = parse_fonts(g)
    out_dir = OUT / "fonts"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for f in fonts:
        name = f["name"]
        src_glyphs = EXTRACTED / "fonts" / f"{name}_glyphs"
        dst = out_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        # copy atlas if present
        atlas = EXTRACTED / "fonts" / f"{name}.png"
        if atlas.exists():
            shutil.copy2(atlas, dst / "atlas.png")
        glyphs_meta = []
        for gl in f["glyphs"]:
            ch = chr(gl["char"]) if 0 <= gl["char"] < 0x110000 else "?"
            entry = {
                "char": ch,
                "codepoint": gl["char"],
                "x": gl["x"],
                "y": gl["y"],
                "w": gl["w"],
                "h": gl["h"],
                "shift": gl["shift"],
                "offset": gl["offset"],
            }
            src = src_glyphs / f"U+{gl['char']:04X}.png"
            if src.exists():
                rel = f"glyphs/{safe_filename(ch)}.png"
                shutil.copy2(src, dst / Path(rel).name)
                # keep unicode name too
                shutil.copy2(src, dst / f"U+{gl['char']:04X}.png")
                entry["file"] = f"U+{gl['char']:04X}.png"
            glyphs_meta.append(entry)
        meta = {
            "name": name,
            "display_name": f.get("display_name"),
            "kind": "font_resource",
            "glyph_count": len(glyphs_meta),
            "glyphs": glyphs_meta,
        }
        (dst / "metrics.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        catalog.append(
            {
                "name": name,
                "display_name": f.get("display_name"),
                "kind": "font_resource",
                "glyphs": len(glyphs_meta),
                "path": f"fonts/{name}",
            }
        )
        print(f"font {name}: {len(glyphs_meta)} glyphs")
    return catalog


def export_sprite_fonts(strings: list[str]) -> list[dict]:
    out_dir = OUT / "sprite_fonts"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    seen_sprites: set[str] = set()

    for var_name in PRIMARY_SPRITE_FONTS:
        sprite, map_idx = SPRITE_FONT_MAP[var_name]
        charset = strings[map_idx]
        src_dir = EXTRACTED / "sprites" / sprite
        if not src_dir.exists():
            print(f"missing sprite {sprite}")
            continue
        frames = sorted(
            src_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1])
        )
        dst = out_dir / var_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        mapped = []
        n = min(len(charset), len(frames))
        for i in range(n):
            ch = charset[i]
            frame = frames[i]
            fname = f"{i:03d}_{safe_filename(ch)}.png"
            shutil.copy2(frame, dst / fname)
            im = Image.open(frame)
            mapped.append(
                {
                    "index": i,
                    "char": ch,
                    "codepoint": ord(ch),
                    "file": fname,
                    "w": im.size[0],
                    "h": im.size[1],
                }
            )

        # also copy sheet if present
        sheet = src_dir / "sheet.png"
        if sheet.exists():
            shutil.copy2(sheet, dst / "sheet.png")

        unused = len(frames) - n
        meta = {
            "name": var_name,
            "sprite": sprite,
            "kind": "sprite_font",
            "charset": charset,
            "charset_len": len(charset),
            "frame_count": len(frames),
            "mapped": n,
            "unmapped_trailing_frames": unused,
            "glyphs": mapped,
            "notes": (
                "Mapped via scr_loadSpriteFonts charset adjacency in STRG. "
                "Trailing frames beyond charset length are left unmapped."
            ),
        }
        (dst / "metrics.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        catalog.append(
            {
                "name": var_name,
                "sprite": sprite,
                "kind": "sprite_font",
                "glyphs": n,
                "frame_count": len(frames),
                "path": f"sprite_fonts/{var_name}",
                "primary_combat": var_name.startswith("fontCeltic")
                or var_name.startswith("fontAlagard"),
            }
        )
        seen_sprites.add(sprite)
        print(f"sprite font {var_name}: {n}/{len(frames)} mapped from {sprite}")

    return catalog


def write_registry(font_cat: list[dict], sprite_cat: list[dict], strings: list[str]) -> None:
    # Collect unique codepoints used by combat fonts
    combat_chars: set[str] = set()
    for var in ("fontCeltic", "fontCeltic2x", "fontAlagard", "fontCelticDemon"):
        charset = strings[SPRITE_FONT_MAP[var][1]]
        combat_chars.update(charset)
    combat_chars.add(" ")

    registry = {
        "game": "The Textorcist: The Story of Ray Bibbia",
        "source": str(DATA_WIN),
        "tooling": {
            "rea": "https://github.com/morluto/rea (RE agent toolkit; PE/analysis)",
            "extractor": "gm-extract (GameMaker data.win FORM parser)",
        },
        "notes": [
            "Combat typing uses sprite fonts loaded in scr_loadSpriteFonts via font_add_sprite_ext.",
            "Normal phrases: fontCeltic / fontCeltic2x (+ outline variants).",
            "Demon phrases: fontCelticDemon (+ outline variants).",
            "Alagard/Gibberesque/Romulus/Easy are UI / alternate text styles.",
            "TrueType FONT chunk fonts (fnt_*) are also present; combat UI primarily uses sprite fonts.",
            "Typed progress is drawn by splitting spelledword/spelledlen (see scr_processTypedNormal).",
            "Ink colors are applied at draw time (draw_set_colour), not baked into glyph sprites.",
        ],
        "scripts_of_interest": [
            "scr_loadSpriteFonts",
            "scr_processTypedNormal",
            "scr_processTypedDemon",
            "scr_drawPhraseNormal",
            "scr_drawPhraseDemon",
            "scr_drawSpelledPhraseNormal",
            "scr_drawSpelledPhraseDemon",
            "scr_drawRayHeadWordNormal",
            "scr_drawRayHeadWordDemon",
            "scr_heroGetPhrase",
            "scr_heroUpdatePhrase",
            "scr_randomWord",
        ],
        "objects_of_interest": [
            "obj_bible",
            "obj_correct_word",
            "obj_mispelled_key",
            "obj_trinket_typer",
        ],
        "state_vars": [
            "currentWord",
            "spelledword",
            "spelledlen",
            "spelledhidden",
        ],
        "combat_charset": "".join(sorted(combat_chars, key=lambda c: ord(c))),
        "combat_charset_len": len(combat_chars),
        "fonts": font_cat,
        "sprite_fonts": sprite_cat,
        "recommended_templates": [
            {
                "role": "combat_normal",
                "path": "sprite_fonts/fontCeltic2x",
                "fallback": "sprite_fonts/fontCeltic",
            },
            {
                "role": "combat_demon",
                "path": "sprite_fonts/fontCelticDemon",
            },
            {
                "role": "ui_gothic",
                "path": "sprite_fonts/fontAlagard",
            },
            {
                "role": "font_resource_celtic",
                "path": "fonts/fnt_celtic_8",
            },
            {
                "role": "font_resource_alagard",
                "path": "fonts/fnt_alagard_16",
            },
        ],
    }
    (OUT / "registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = """# Textorcist glyph pack

Extracted from `data.win` (GameMaker FORM) using **gm-extract**, with sprite-font
character maps recovered from `scr_loadSpriteFonts` / `font_add_sprite_ext` strings.

## Layout

- `fonts/` — FONT-chunk bitmaps (`fnt_*`, `font5`) with `U+XXXX.png` + `metrics.json`
- `sprite_fonts/` — gameplay sprite fonts remapped to characters (`000_U+0061_a.png`, …)
- `registry.json` — catalog, combat charset, scripts/objects of interest

## Recommended OCR templates

| Role | Pack |
|------|------|
| Combat (normal) | `sprite_fonts/fontCeltic2x` (fallback `fontCeltic`) |
| Combat (demon) | `sprite_fonts/fontCelticDemon` |
| Gothic UI | `sprite_fonts/fontAlagard` |

Glyph sprites are white-on-transparent/black masks. In-game ink (brown remaining,
black/blue typed) is a **draw tint**, so template match on a binary/ink mask of the
parchment ROI, not raw RGB.

## Related project data

- `phrases.json` / `lines.json` / `words.json` — dialogue & combat phrase corpus
- `hop_phrases.json` — multi-word hop targets (Flee / Away Ye Hostiles, …)
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if not EXTRACTED.exists():
        raise SystemExit(f"missing {EXTRACTED}; run gm-extract first")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    strings = load_strings()
    font_cat = export_font_metrics()
    sprite_cat = export_sprite_fonts(strings)
    write_registry(font_cat, sprite_cat, strings)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
