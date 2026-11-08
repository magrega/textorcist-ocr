from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from auto_roi import find_parchment
from parchment_reader import read_parchment


class AutoRoiTests(unittest.TestCase):
    def test_detects_and_reads_at_multiple_resolutions(self):
        source = cv2.imread(str(Path(__file__).parent / "debug_capture.png"))
        self.assertIsNotNone(source, "Keep debug_capture.png as the auto-capture fixture.")
        for size in [(3840, 2160), (2560, 1440), (1920, 1080), (1366, 768), (3840, 1200)]:
            with self.subTest(size=size):
                frame = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
                rect = find_parchment(frame)
                self.assertIsNotNone(rect)
                x, y, width, height = rect
                self.assertAlmostEqual((x + width / 2) / size[0], 0.5, delta=0.04)
                self.assertGreater(y, size[1] * 0.7)
                self.assertGreater(width / height, 6)
                self.assertIsNotNone(read_parchment(frame[y:y + height, x:x + width]))

    def test_rejects_non_battle_frames(self):
        self.assertIsNone(find_parchment(np.zeros((1080, 1920, 3), np.uint8)))
        tiny = np.zeros((1080, 1920, 3), np.uint8)
        tiny[900:1000, 700:1200] = (184, 200, 216)
        self.assertIsNone(find_parchment(tiny))


if __name__ == "__main__":
    unittest.main()
