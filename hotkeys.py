"""System-wide hotkeys through Win32 RegisterHotKey.

Unlike a low-level keyboard hook, RegisterHotKey cannot miss a key release
while the process is busy, so a hotkey never gets stuck in a "held" state.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading
from typing import Callable

MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
VK_CODES = {f"f{n}": 0x6F + n for n in range(1, 13)}

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class HotKeys:
    def __init__(self, bindings: dict[str, Callable[[], None]]):
        self.bindings = list(bindings.items())
        self.failed: list[str] = []
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread_id = 0
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(2)

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._thread.join(2)

    def _run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        registered = []
        for index, (name, _) in enumerate(self.bindings, start=1):
            if user32.RegisterHotKey(None, index, MOD_NOREPEAT, VK_CODES[name.lower()]):
                registered.append(index)
            else:
                self.failed.append(name.upper())
        self._ready.set()
        msg = wt.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and 1 <= msg.wParam <= len(self.bindings):
                    self.bindings[msg.wParam - 1][1]()
        finally:
            for index in registered:
                user32.UnregisterHotKey(None, index)
