# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['helper.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('glyphs/sprite_fonts/fontAlagard', 'glyphs/sprite_fonts/fontAlagard'),
        ('words.json', '.'),
        ('hop_phrases.json', '.'),
        ('lines.json', '.'),
        ('assets/app.ico', 'assets'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6.QtWebEngine', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtNetwork', 'PySide6.QtOpenGL',
        'PySide6.QtSvg', 'PySide6.QtQml', 'PySide6.QtQuick',
        # Legacy OCR tooling and GUI toolkits the helper never imports.
        'PIL', 'pytesseract', 'pygetwindow', 'tkinter', '_tkinter',
        # No network access: skip OpenSSL (hashlib falls back to built-in hashes).
        'ssl', '_ssl', '_hashlib',
    ],
    noarchive=False,
    optimize=0,
)


def _name(entry):
    return entry[0].replace('\\', '/').lower()


QT_PLUGINS_KEPT = {
    'pyside6/plugins/platforms/qwindows.dll',
    'pyside6/plugins/styles/qmodernwindowsstyle.dll',
    'pyside6/plugins/imageformats/qico.dll',
}
# Only reached through plugins removed above (QML, virtual keyboard, PDF, TLS, SVG)
# or as a fallback renderer for machines without OpenGL drivers.
QT_DLLS_DROPPED = {
    'opengl32sw.dll', 'qt6pdf.dll', 'qt6quick.dll', 'qt6qml.dll', 'qt6qmlmodels.dll',
    'qt6qmlmeta.dll', 'qt6qmlworkerscript.dll', 'qt6virtualkeyboard.dll',
    'qt6network.dll', 'qt6svg.dll', 'qt6opengl.dll',
}


def _keep(entry):
    name = _name(entry)
    if name.startswith('cv2/opencv_videoio_ffmpeg'):
        return False  # the app only reads screenshots, never video
    if name.startswith('pyside6/plugins/'):
        return name in QT_PLUGINS_KEPT
    if name.startswith('pyside6/translations/'):
        return False
    return not (name.startswith('pyside6/') and name.split('/')[-1] in QT_DLLS_DROPPED)


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

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
