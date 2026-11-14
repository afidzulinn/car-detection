import json
import sys
import tempfile
from pathlib import Path
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_collection"))
from verify_dataset import audit_dataset


class TestVerifyDataset(unittest.TestCase):
    def test_audit_dataset_detects_identical_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            img = Image.new("RGB", (1280, 720), (220, 220, 220))
            img.save(p / "sample_00000.jpg")
            img.save(p / "sample_00001.jpg")
            ann = [
                {
                    "file": "sample_00000.jpg",
                    "labels": {k: 0 for k in ("front_left_door", "front_right_door", "rear_left_door", "rear_right_door", "hood")},
                    "angle": 0,
                    "elevation": "mid",
                },
                {
                    "file": "sample_00001.jpg",
                    "labels": {k: 1 for k in ("front_left_door", "front_right_door", "rear_left_door", "rear_right_door", "hood")},
                    "angle": 0,
                    "elevation": "mid",
                },
            ]
            (p / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
            report = audit_dataset(p)
            self.assertFalse(report["passed"])
            self.assertEqual(report["min_open_closed_mae_by_elev"]["mid"], 0.0)


if __name__ == "__main__":
    unittest.main()
