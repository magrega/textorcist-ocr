"""Find and capture the Textorcist game window by process name.

Handles Windows DPI scaling (e.g. 4K monitor at 150%/200%) so mss regions
match the real on-screen pixels instead of grabbing only a corner.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

import cv2
import mss
import numpy as np

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DPI_CONTEXT_V2 = ctypes.c_void_p(-4)

_dpi_ready = False


def enable_dpi_awareness() -> None:
    global _dpi_ready
    if _dpi_ready:
        return
    try:
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(_DPI_CONTEXT_V2)
        else:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    _dpi_ready = True


enable_dpi_awareness()


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


@dataclass
class GameWindow:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int
    minimized: bool
    foreground: bool

    def mss_region(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": max(1, self.width),
            "height": max(1, self.height),
        }


def _process_basename(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.rsplit("\\", 1)[-1]
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _physical_client_box(hwnd: int) -> tuple[int, int, int, int]:
    enable_dpi_awareness()

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")

    points = (wintypes.POINT * 2)()
    points[0] = wintypes.POINT(rect.left, rect.top)
    points[1] = wintypes.POINT(rect.right, rect.bottom)
    user32.MapWindowPoints(hwnd, None, ctypes.byref(points), 2)

    left, top = points[0].x, points[0].y
    right, bottom = points[1].x, points[1].y
    width, height = right - left, bottom - top

    try:
        dwm = ctypes.windll.dwmapi
        frame = RECT()
        hr = dwm.DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(frame),
            ctypes.sizeof(frame),
        )
        if hr == 0:
            frame_w = frame.right - frame.left
            frame_h = frame.bottom - frame.top
            if frame_w > width * 1.2 or frame_h > height * 1.2:
                left, top, width, height = frame.left, frame.top, frame_w, frame_h
    except Exception:
        pass

    return left, top, width, height


def find_game_window(
    process_name: str = "Textorcist.exe",
    title_substr: str = "Textorcist",
) -> GameWindow:
    """Find the game HWND even if minimized/hidden (GameMaker focus-loss)."""
    enable_dpi_awareness()
    matches: list[tuple[int, str, tuple[int, int, int, int], bool]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd: int, _: int) -> bool:
        if not user32.IsWindow(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if _process_basename(pid.value).lower() != process_name.lower():
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        # Prefer real game windows (titled); skip tiny helper HWNDs
        try:
            left, top, width, height = _physical_client_box(hwnd)
        except Exception:
            return True
        if width < 100 or height < 100:
            return True
        if title_substr.lower() not in title.lower() and length > 0:
            # Still keep untitled large windows from this process as fallback
            pass

        minimized = bool(user32.IsIconic(hwnd)) or not bool(user32.IsWindowVisible(hwnd))
        matches.append((hwnd, title, (left, top, width, height), minimized))
        return True

    user32.EnumWindows(enum_proc, 0)
    if not matches:
        raise RuntimeError(
            f"Could not find {process_name}. Is the game running?"
        )

    # Prefer titled + visible, then titled, then anything
    def rank(item: tuple) -> tuple:
        hwnd, title, box, minimized = item
        return (
            0 if title_substr.lower() in title.lower() else 1,
            1 if minimized else 0,
            -(box[2] * box[3]),
        )

    matches.sort(key=rank)
    hwnd, title, (left, top, width, height), minimized = matches[0]
    foreground = user32.GetForegroundWindow() == hwnd
    return GameWindow(
        hwnd=hwnd,
        title=title or process_name,
        left=left,
        top=top,
        width=width,
        height=height,
        minimized=minimized,
        foreground=foreground,
    )


def focus_window(hwnd: int) -> None:
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def grab_game_frame(
    process_name: str = "Textorcist.exe",
    title_substr: str = "Textorcist",
    focus: bool = False,
) -> tuple[np.ndarray, GameWindow]:
    enable_dpi_awareness()
    win = find_game_window(process_name, title_substr)
    if focus:
        focus_window(win.hwnd)
        win = find_game_window(process_name, title_substr)
    with mss.MSS() as sct:
        mon = sct.monitors[0]  # virtual desktop
        print(
            f"[capture] window=({win.left},{win.top}) {win.width}x{win.height} | "
            f"virtual={mon['width']}x{mon['height']} | "
            f"minimized={win.minimized} foreground={win.foreground}"
        )
        shot = np.array(sct.grab(win.mss_region()))
    bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    return bgr, win


def grab_window_bgr(win: GameWindow, sct: mss.MSS) -> np.ndarray:
    """Grab full game window; caller crops ROI in image space."""
    shot = np.array(sct.grab(win.mss_region()))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
