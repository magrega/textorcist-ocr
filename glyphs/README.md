# Textorcist glyph pack

Extracted from `data.win` (GameMaker FORM) using **gm-extract**, with sprite-font
character maps recovered from `scr_loadSpriteFonts` / `font_add_sprite_ext` strings.

## Layout

- `sprite_fonts/fontAlagard` — the sprite font drawn on the battle parchment,
  remapped to characters (`000_U+0061_a.png`, …) with `metrics.json`
- `game_info.json` — notes on the game's typing scripts and font variables

`scripts/build_glyph_pack.py` regenerates this folder from `extracted/`, including
the other sprite fonts and FONT-chunk bitmaps that the helper does not use.

Glyph sprites are white-on-transparent masks. In-game ink (brown remaining,
navy typed) is a draw tint, so match against a binary ink mask of the parchment,
not raw RGB.
