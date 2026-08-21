import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    from create_pupil_masks import convert_annotations, render_mask
except ImportError as exc:
    convert_annotations = None
    render_mask = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class RenderMaskTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"create_pupil_masks is not implemented: {IMPORT_ERROR}")

    def test_ellipse_region_produces_binary_filled_mask(self):
        regions = [{
            "shape_attributes": {
                "name": "ellipse",
                "cx": 20,
                "cy": 15,
                "rx": 8,
                "ry": 4,
                "theta": 0,
            }
        }]

        mask = render_mask((30, 40), regions)

        self.assertEqual(mask[15, 20], 255)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_polyline_region_is_closed_and_filled(self):
        regions = [{
            "shape_attributes": {
                "name": "polyline",
                "all_points_x": [10, 20, 20, 10],
                "all_points_y": [10, 10, 20, 20],
            }
        }]

        mask = render_mask((30, 40), regions)

        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(mask[5, 5], 0)

    def test_no_regions_produces_black_blink_mask(self):
        mask = render_mask((30, 40), [])

        self.assertEqual(np.count_nonzero(mask), 0)


class ConvertAnnotationsTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"create_pupil_masks is not implemented: {IMPORT_ERROR}")

    def test_conversion_writes_one_png_mask_per_json_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            output_dir = root / "masks"
            image_dir.mkdir()

            image = np.zeros((24, 32), dtype=np.uint8)
            cv2.imwrite(str(image_dir / "image_1.jpg"), image)
            cv2.imwrite(str(image_dir / "image_2.jpg"), image)

            annotations = {
                "image_1.jpg1": {
                    "filename": "image_1.jpg",
                    "regions": [{
                        "shape_attributes": {
                            "name": "ellipse",
                            "cx": 16,
                            "cy": 12,
                            "rx": 5,
                            "ry": 3,
                            "theta": 0,
                        }
                    }],
                },
                "image_2.jpg1": {
                    "filename": "image_2.jpg",
                    "regions": [],
                },
            }
            json_path = root / "labels.json"
            json_path.write_text(json.dumps(annotations), encoding="utf-8")

            summary = convert_annotations(json_path, image_dir, output_dir)

            self.assertEqual(summary, {"total": 2, "pupil": 1, "blink": 1})
            self.assertTrue((output_dir / "image_1.png").exists())
            blink_mask = cv2.imread(
                str(output_dir / "image_2.png"), cv2.IMREAD_GRAYSCALE
            )
            self.assertEqual(np.count_nonzero(blink_mask), 0)


if __name__ == "__main__":
    unittest.main()
