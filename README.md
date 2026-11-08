# Textorcist OCR helper

A local Windows desktop assistant for The Textorcist. It reads the battle parchment and types its remaining letters, with a separate save-file import tool.

## Stack

- **Python 3.10** — application, capture, OCR, and save processing.
- **PySide6 / Qt 6** — desktop UI.
- **OpenCV and NumPy** — image analysis, automatic parchment detection, and sprite matching.
- **MSS** — screenshots of the game window.
- **pynput** — keyboard input.
- **Win32 RegisterHotKey** (`hotkeys.py`) — global F6/F8 hotkeys in the helper app.
- **RapidFuzz** — limited vocabulary correction for uncertain glyphs.
- **PyInstaller** — standalone Windows executable.
- **Standard-library JSON and Base64 tools** — settings and save conversion.

The bot's main recognition path uses the game's sprite font, not Tesseract. Tesseract is needed only for the separate legacy OCR scripts. The app is Windows-specific and does not use React, a web server, a database, or a cloud service.

## Run the executable

Run:

```powershell
.\dist\"Textorcist OCR helper.exe"
```

It includes the Python runtime, Qt, the OCR engine, and font assets. A separate Python or Tesseract installation is not needed. The parchment-and-cross icon is used for the application and taskbar.

### Bot

Automatic parchment detection is on by default. Start the game in windowed/borderless mode, enter a battle, click **Start typing**, and focus the game. **F8** toggles typing globally. Typing pauses whenever the game is not focused.

The detector searches the lower portion of the game for the beige parchment, checks its geometry and center position, and scales coordinates to the actual game-window size. It rechecks when that size changes. No screen-resolution calibration is required.

If the parchment is not found, use **Manual capture instead…**: focus the game, press **F6**, drag around the bottom parchment text on the frozen screenshot, and save. A manual selection takes precedence over automatic detection. To restore automatic detection, edit the settings file described below and set `"auto_roi": true`.

### Save import

Close the game completely. If Steam Cloud is enabled, pause it for the import so it does not overwrite local changes.

Choose the incoming `.dat` and corresponding local save—normally `progress.dat` in `%LOCALAPPDATA%\Textorcist`. After confirmation, the app preserves the current installation's binary prefix and trailing bytes and replaces only its JSON progress.

All `.dat` files in the destination folder are backed up to a unique subfolder of `helper-backups` before the selected save is replaced. The app refuses to import while the game is running. To undo an import, close the game and copy the backup `.dat` files back into the save folder. Save processing stays entirely local.

## Manual development setup

Requires 64-bit Windows and Python 3.10. From the project directory:

```powershell
cd D:\react-projects\textorcist-ocr
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python helper.py
```

The packaged executable uses `%LOCALAPPDATA%\TextorcistOCRHelper` for writable settings and debug images. When run from Python, settings and debug images stay in the project directory. Do not edit files inside PyInstaller's temporary `_MEI...` directory; those files are removed after exit.

### Run the command-line bot

```powershell
python bot.py
```

The command-line bot uses the saved `roi` in `config.json`. **F8** toggles typing, **F7** saves a debug snapshot, and **F9** quits. Keep the game focused.

Manual ROI calibration remains available:

```powershell
python calibrate.py
```

Focus the game, press **F6**, draw the parchment box, and press Enter. Alternatively, supply a screenshot: `python calibrate.py path\to\shot.png`.

## Build the executable manually

With Python 3.10 installed, run the build script from any directory:

```powershell
& "D:\react-projects\textorcist-ocr\rebuild.ps1"
```

The script installs dependencies and PyInstaller, runs the automatic-capture and save-import tests, and builds the executable. It uses `.venv` when present, otherwise `py -3.10`. Optional flags: `-SkipDependencies` to retain installed packages and `-SkipTests` to omit tests.

To perform the same build manually with a virtual environment activated:

```powershell
python -m pip install pyinstaller
pyinstaller --noconfirm --clean "Textorcist OCR helper.spec"
```

The spec file bundles the font assets, dictionaries, and `assets\app.ico`. It builds:

```text
dist\Textorcist OCR helper.exe
```

Close a running copy before rebuilding. `build/` is intermediate build output and is not required to run the application.

## Tests and diagnostics

```powershell
python -m unittest -v test_auto_roi.py test_save_import.py
python scripts\sim_bot.py
```

The automatic-capture tests cover multiple resolutions and reject frames without a valid parchment. Save tests use temporary files only.

For troubleshooting, inspect the saved debug PNGs. Lower `key_delay_ms` for faster typing or increase it if the game drops keystrokes. `stall_ms` controls how long the bot waits before retrying a key. Keep the game focused and avoid overlapping its window.
