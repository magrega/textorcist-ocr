# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['helper.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('glyphs', 'glyphs'),
        ('words.json', '.'),
        ('hop_phrases.json', '.'),
        ('lines.json', '.'),
        ('assets/app.ico', 'assets'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngine', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets'],
    noarchive=False,
    optimize=0,
)


def _name(entry):
    return entry[0].replace('\\', '/').lower()


# The app only reads screenshots; OpenCV's video I/O backend is never used.
a.binaries = [b for b in a.binaries if not _name(b).startswith('cv2/opencv_videoio_ffmpeg')]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Textorcist OCR helper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\app.ico'],
)
