"""Textorcist OCR helper — a local desktop interface for the OCR bot and saves."""
from __future__ import annotations

from window_capture import enable_dpi_awareness, find_game_window, grab_game_frame, grab_window_bgr
enable_dpi_awareness()

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import cv2
import mss
import numpy as np
from PySide6.QtCore import Qt, QRect, QThread, QTimer, Signal, QEvent
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)
from auto_roi import find_parchment
from bot import Bot, CONFIG_PATH
from glyph_matcher import load_primary_pack
from hotkeys import HotKeys
from save_import import default_save_directory, import_folder, prepare_folder_import


class CaptureDialog(QDialog):
    def __init__(self, bgr, parent):
        super().__init__(parent)
        self.setWindowTitle("Select parchment text")
        self.setModal(True)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.picture = QPixmap.fromImage(image)
        self.picture.setDevicePixelRatio(1)
        screen = parent.screen().availableGeometry()
        self.scale = min(1, (screen.width() - 120) / image.width(), (screen.height() - 180) / image.height())
        self.selection = None
        self.origin = None
        self.saved = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Drag a box around the bottom parchment text. Include the whole text line."))
        self.view = QLabel()
        self.view.setMinimumSize(min(image.width(), int(image.width() * self.scale)), min(image.height(), int(image.height() * self.scale)))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setCursor(Qt.CrossCursor)
        layout.addWidget(self.view)
        self.view.installEventFilter(self)
        buttons = QHBoxLayout()
        buttons.addStretch()
        save = QPushButton("Save area")
        save.setDefault(True)
        save.clicked.connect(self.save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self.redraw()

    def eventFilter(self, obj, event):
        if obj is self.view:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.origin = self.pos(event.position().toPoint())
                self.selection = QRect(self.origin, self.origin)
                self.redraw()
                return True
            if event.type() == QEvent.MouseMove and self.origin is not None:
                self.selection = QRect(self.origin, self.pos(event.position().toPoint())).normalized()
                self.redraw()
                return True
        return super().eventFilter(obj, event)

    def pos(self, point):
        size = self.picture.size() * self.scale
        offset = (self.view.size() - size) / 2
        return point - offset.toPoint()

    def redraw(self):
        canvas = self.picture.copy()
        if self.selection is not None:
            painter = QPainter(canvas)
            painter.setPen(QPen(QColor("#ffad42"), max(2, round(3 / self.scale))))
            painter.drawRect(self.selection)
            painter.end()
        self.view.setPixmap(canvas.scaled(canvas.size() * self.scale, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.view.adjustSize()

    def save(self):
        if self.selection is None:
            QMessageBox.warning(self, "Select an area", "Drag around the parchment text first.")
            return
        rect = self.selection.intersected(QRect(self.picture.rect()))
        if rect.width() / self.scale < 10 or rect.height() / self.scale < 10:
            QMessageBox.warning(self, "Area too small", "Select a larger parchment text area.")
            return
        self.saved = True
        self.area = rect
        self.accept()


class BotWorker(QThread):
    status = Signal(str)
    roi_status = Signal(str)
    failed = Signal(str)

    def __init__(self, bot, stop_event):
        super().__init__()
        self.bot, self.stop_event = bot, stop_event
        self.auto_roi = None
        self.auto_roi_size = None
        self.last_detect = 0.0

    def detect_parchment(self, frame, win):
        size = (win.width, win.height)
        if self.auto_roi is not None and size == self.auto_roi_size:
            return
        if self.auto_roi is None and size == self.auto_roi_size and time.monotonic() - self.last_detect < 0.5:
            return
        self.last_detect = time.monotonic()
        found = find_parchment(frame)
        if found != self.auto_roi or size != self.auto_roi_size:
            if found is None:
                self.roi_status.emit(f"Automatic capture: searching for the parchment in the {win.width}×{win.height} game window…")
            else:
                x, y, width, height = found
                self.roi_status.emit(f"Automatic capture: parchment found at x={x}, y={y}, {width}×{height} px (game window {win.width}×{win.height}).")
        self.auto_roi, self.auto_roi_size = found, size

    def run(self):
        try:
            with mss.MSS() as capture:
                while not self.stop_event.wait(float(self.bot.cfg.get("poll_ms", 50)) / 1000):
                    if not self.bot.enabled:
                        continue
                    try:
                        win = find_game_window(self.bot.cfg.get("process_name", "Textorcist.exe"), self.bot.cfg.get("window_title_substr", "Textorcist"))
                        if win.minimized or not win.foreground:
                            self.status.emit("Typing ON · Paused — focus the game")
                            self.stop_event.wait(0.2)
                            continue
                        if self.bot.cfg.get("auto_roi", True):
                            frame = grab_window_bgr(win, capture)
                            self.detect_parchment(frame, win)
                            if self.auto_roi is None:
                                self.status.emit("Parchment not found · Open a battle or use manual capture")
                                self.stop_event.wait(0.2)
                                continue
                            x, y, width, height = self.auto_roi
                            roi = frame[y:y + height, x:x + width]
                        else:
                            roi = cv2.cvtColor(np.array(capture.grab(self.bot.roi_region(win))), cv2.COLOR_BGRA2BGR)
                        self.bot.process_frame(roi)
                        self.status.emit("Typing ON · " + (self.bot._line_text or "Waiting for parchment text"))
                    except Exception as exc:
                        self.status.emit(str(exc))
                        self.stop_event.wait(0.5)
        except Exception as exc:
            self.failed.emit(str(exc))


class Helper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Textorcist OCR helper")
        self.resize(880, 780)
        self.setMinimumSize(720, 640)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.capture_requested = False
        self.selecting = False
        self.busy = False
        self.bot = Bot()
        self.worker = None
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self.bot_tab(), "Bot")
        tabs.addTab(self.save_tab(), "Save import")
        self.setStyleSheet("""
            QMainWindow, QDialog { background: #17191e; color: #ece7dc; }
            QWidget { color: #ece7dc; font: 14px 'Segoe UI'; }
            QTabWidget::pane { border: 0; }
            QTabBar::tab { padding: 14px 22px; color: #e4c69c; background: #22242b; }
            QTabBar::tab:hover { color: #ffe0af; background: #2b2d35; }
            QTabBar::tab:selected { color: #ffc978; background: #2b2925; border-bottom: 2px solid #e8a64b; }
            QPushButton { background: #0d0d0f; color: #e5a94f; padding: 11px 17px; border: 1px solid #e5a94f; border-radius: 4px; font-weight: 600; }
            QPushButton:hover { background: #1f1a12; color: #ffc978; border-color: #ffc978; }
            QPushButton:pressed { background: #2b2316; }
            QPushButton[on="true"] { background: #e5a94f; color: #0d0d0f; }
            QPushButton[on="true"]:hover { background: #ffc978; color: #0d0d0f; }
            QPushButton#toggleButton { font-size: 16px; padding: 15px 17px; }
            QPushButton:disabled { background: #2a2a2d; color: #77736d; border-color: #4a4a4f; }
            QLineEdit { padding: 10px; background: #242831; border: 1px solid #3b414d; }
            QLineEdit:focus { border-color: #e5a94f; }
        """)
        self.hotkeys = HotKeys({"f8": lambda: self.events.put(("toggle", None)), "f6": lambda: self.events.put(("capture", None))})
        self.hotkeys.start()
        if self.hotkeys.failed:
            self.status.setText(f"{' and '.join(self.hotkeys.failed)} is used by another program. Use the buttons instead.")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(80)

    def label(self, text, parent, heading=False):
        label = QLabel(text, parent)
        label.setWordWrap(True)
        if heading:
            label.setStyleSheet("font-size: 22px; font-weight: 600; color: #ffc978;")
        return label

    def set_on(self, button, on):
        if button.property("on") != on:
            button.setProperty("on", on)
            button.style().unpolish(button)
            button.style().polish(button)

    def button(self, text, command):
        button = QPushButton(text)
        button.setProperty("on", False)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(command)
        return button

    def bot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)
        layout.addWidget(self.label("Type the words. Keep the adventure.", tab, True))
        layout.addWidget(self.label("A local screen reader and typing assistant for The Textorcist.", tab))
        layout.addSpacing(16)
        layout.addWidget(self.label("Parchment detection", tab))
        layout.addWidget(self.label("Automatic capture finds the beige parchment near the bottom center of the game. It works across resolutions and window sizes; no setup is normally needed.", tab))
        self.capture_button = self.button("Manual capture instead…", self.arm_capture)
        self.capture_button.setFixedWidth(260)
        layout.addWidget(self.capture_button)
        self.capture_status = self.label("Ready for automatic detection. Manual capture is available if the parchment is not found.", tab)
        layout.addWidget(self.capture_status)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #353b47;")
        layout.addWidget(line)
        layout.addWidget(self.label("Typing control", tab))
        layout.addWidget(self.label("Start here and switch to the game. F8 toggles typing from anywhere. Typing pauses whenever the game is not focused.", tab))
        self.toggle_button = self.button("Start typing  ·  F8", self.toggle)
        self.toggle_button.setFixedWidth(260)
        self.toggle_button.setObjectName("toggleButton")
        layout.addWidget(self.toggle_button)
        self.status = self.label("Typing OFF", tab)
        layout.addWidget(self.status)
        layout.addStretch()
        return tab

    def folder_row(self, parent, title, initial):
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.addWidget(self.label(title, parent))
        row = QHBoxLayout()
        field = QLineEdit(initial)
        row.addWidget(field)
        def browse():
            path = QFileDialog.getExistingDirectory(self, title, field.text() or str(default_save_directory()))
            if path:
                field.setText(path)
        row.addWidget(self.button("Browse…", browse))
        layout.addLayout(row)
        return field, layout

    def save_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)
        layout.addWidget(self.label("Bring your progress with you", tab, True))
        layout.addWidget(self.label("Close the game completely. If Steam Cloud is enabled, pause it for this import so it does not overwrite local changes.\n\nLaunch Textorcist once on this PC so it creates local .dat files. Then choose the folder with your old saves. The helper keeps each local file's binary prefix and replaces only the JSON progress inside every matching .dat.", tab))
        self.source, source_layout = self.folder_row(tab, "Incoming save folder", "")
        self.target, target_layout = self.folder_row(tab, "Current local save folder", str(default_save_directory()))
        layout.addLayout(source_layout)
        layout.addLayout(target_layout)
        layout.addWidget(self.label("Matching is by filename: progress.dat → progress.dat, ints.dat → ints.dat, and so on. All local .dat files are backed up first.", tab))
        self.import_button = self.button("Back up and import…", self.confirm_import)
        self.import_button.setFixedWidth(260)
        layout.addWidget(self.import_button)
        self.save_status = self.label("No files changed.", tab)
        layout.addWidget(self.save_status)
        layout.addWidget(self.button("Open save folder / backups", self.open_save_folder))
        layout.addStretch()
        return tab

    def toggle(self):
        if self.bot.enabled:
            self.stop_typing()
            return
        if self.capture_requested or self.selecting or self.busy:
            return
        if load_primary_pack() is None:
            QMessageBox.critical(self, "Font sprites missing", "Run scripts/build_glyph_pack.py before starting.")
            return
        self.bot.toggle()
        if self.worker is None:
            self.worker = BotWorker(self.bot, self.stop_event)
            self.worker.status.connect(self.status.setText)
            self.worker.roi_status.connect(self.capture_status.setText)
            self.worker.failed.connect(self.worker_failed)
            self.worker.start()
        self.sync_toggle()

    def sync_toggle(self):
        self.toggle_button.setText("Stop typing  ·  F8" if self.bot.enabled else "Start typing  ·  F8")
        self.set_on(self.toggle_button, self.bot.enabled)
        self.status.setText("Typing ON — switch to the game." if self.bot.enabled else "Typing OFF")
        self.refresh_buttons()

    def refresh_buttons(self):
        self.capture_button.setText("Cancel capture" if self.capture_requested else "Manual capture instead…")
        self.set_on(self.capture_button, self.capture_requested)
        self.toggle_button.setEnabled(self.bot.enabled or not self.capture_requested)
        self.import_button.setEnabled(not self.capture_requested)

    def stop_typing(self):
        self.bot.enabled = False
        self.sync_toggle()

    def arm_capture(self):
        if self.busy or self.selecting:
            return
        self.stop_typing()
        self.capture_requested = not self.capture_requested
        self.refresh_buttons()
        self.capture_status.setText("Switch to the game, then press F6 to freeze a screenshot." if self.capture_requested else "Capture cancelled. Previous area kept.")

    def capture(self):
        if not self.capture_requested:
            return
        try:
            win = find_game_window(self.bot.cfg.get("process_name", "Textorcist.exe"), self.bot.cfg.get("window_title_substr", "Textorcist"))
            if win.minimized or not win.foreground:
                raise ValueError("Focus the game first, then press F6 again.")
            frame, _ = grab_game_frame(self.bot.cfg.get("process_name", "Textorcist.exe"), self.bot.cfg.get("window_title_substr", "Textorcist"))
            if float(np.mean(frame)) < 5:
                raise ValueError("Capture is nearly black. Focus the visible game and retry F6.")
            self.capture_requested = False
            self.refresh_buttons()
            self.select_area(frame)
        except Exception as exc:
            self.capture_status.setText(str(exc))

    def select_area(self, frame):
        self.selecting = True
        try:
            dialog = CaptureDialog(frame, self)
            if dialog.exec() != QDialog.Accepted or not dialog.saved:
                self.capture_status.setText("Capture cancelled. Previous area kept.")
                return
            rect = dialog.area
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg["auto_roi"] = False
            cfg["roi"] = {
                "rel_left": rect.left() / dialog.picture.width(),
                "rel_top": rect.top() / dialog.picture.height(),
                "rel_right": (rect.right() + 1) / dialog.picture.width(),
                "rel_bottom": (rect.bottom() + 1) / dialog.picture.height(),
            }
            temporary = CONFIG_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            os.replace(temporary, CONFIG_PATH)
            self.bot.cfg = cfg
            self.bot._reset_send_state()
            self.capture_status.setText("Parchment area saved. Switch to the game and press F8 to start typing.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save area", str(exc))
        finally:
            self.selecting = False

    def confirm_import(self):
        if self.busy or self.selecting or self.capture_requested:
            return
        self.stop_typing()
        source, target = Path(self.source.text().strip()), Path(self.target.text().strip())
        try:
            result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Textorcist.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if "textorcist.exe" in result.stdout.lower():
                raise ValueError("Close Textorcist completely before importing saves.")
            planned = prepare_folder_import(source, target)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot import saves", str(exc))
            return
        self.busy = True
        try:
            names = "\n".join(sorted(planned))
            answer = QMessageBox.question(
                self,
                "Back up and import all matching saves?",
                f"Incoming folder:\n{source}\n\nLocal folder:\n{target}\n\n"
                f"{len(planned)} .dat file(s) will be updated:\n{names}\n\n"
                "All local .dat files will be backed up first. Keep the game closed and Steam Cloud paused. Continue?",
            )
            if answer != QMessageBox.Yes:
                return
            backup, written = import_folder(source, target)
            self.save_status.setText(
                f"Import complete: {len(written)} file(s).\nBackup: {backup}\n\n"
                "To undo: close the game and copy the backed-up .dat files into the save folder."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
        finally:
            self.busy = False

    def open_save_folder(self):
        folder = Path(self.target.text().strip() or str(default_save_directory()))
        if folder.is_dir():
            os.startfile(folder)
        else:
            QMessageBox.information(self, "Folder not found", "Launch the game once to create its local saves.")

    def worker_failed(self, error):
        self.bot.enabled = False
        self.worker = None
        self.sync_toggle()
        self.status.setText("Bot stopped: " + error)

    def poll(self):
        for _ in range(100):
            try:
                event, _ = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "toggle":
                self.toggle()
            elif event == "capture":
                self.capture()

    def closeEvent(self, event):
        self.bot.enabled = False
        self.bot.running = False
        self.stop_event.set()
        self.hotkeys.stop()
        event.accept()


def app_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def load_app_icon() -> QIcon:
    icon_path = app_root() / "assets" / "app.ico"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Textorcist OCR helper")
    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = Helper()
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
