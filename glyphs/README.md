# Textorcist glyph pack

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
