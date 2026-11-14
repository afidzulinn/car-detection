import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_collection"))

from collect_data import (
    compute_safe_drag_offset,
    canonical_sample_id,
    parse_args,
)


class TestCollectData(unittest.TestCase):
    def test_compute_safe_drag_offset_stays_in_bounds_and_outside_car(self):
        for w, h in [(1280, 720), (1070, 805), (1024, 600)]:
            ox, oy = compute_safe_drag_offset(w, h)
            x_start, y_start = w / 2 + ox, h / 2 + oy
            # Must be strictly below car ROI (y > 0.62 * h)
            self.assertGreater(y_start, 0.62 * h)
            # Dragging left by -115 and down by +85 must stay inside [0, w] x [0, h]
            self.assertTrue(0 < x_start - 115 < w)
            self.assertTrue(0 < y_start + 85 < h)

    def test_canonical_sample_id_bijective_0_to_767(self):
        ids = set()
        import itertools
        for combo in itertools.product([0, 1], repeat=5):
            for elev in ("mid", "high", "top"):
                for angle in (0, 45, 90, 135, 180, 225, 270, 315):
                    sid = canonical_sample_id(combo, elev, angle)
                    ids.add(sid)
        self.assertEqual(len(ids), 768)
        self.assertEqual(min(ids), 0)
        self.assertEqual(max(ids), 767)
        self.assertEqual(canonical_sample_id((0, 0, 0, 0, 0), "mid", 0), 0)
        self.assertEqual(canonical_sample_id((1, 1, 1, 1, 1), "top", 315), 767)

    def test_parse_args_flags(self):
        args = parse_args(["--pilot", "--fresh", "--headless"])
        self.assertTrue(args.pilot)
        self.assertTrue(args.fresh)
        self.assertTrue(args.headless)


if __name__ == "__main__":
    unittest.main()
