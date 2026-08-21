import unittest

try:
    from prepare_pupil_split import build_split
except ImportError as exc:
    build_split = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class BuildSplitTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"prepare_pupil_split is not implemented: {IMPORT_ERROR}")

    def test_recording_boundary_keeps_each_range_in_one_split(self):
        pupil_region = {"shape_attributes": {"name": "ellipse"}}
        entries = [
            {"filename": "image_1.jpg", "regions": [pupil_region]},
            {"filename": "image_235.jpg", "regions": []},
            {"filename": "image_236.jpg", "regions": [pupil_region]},
            {"filename": "image_300.jpg", "regions": []},
        ]

        manifest = build_split(entries, validation_start=236)

        self.assertEqual(
            [sample["filename"] for sample in manifest["train"]],
            ["image_1.jpg", "image_235.jpg"],
        )
        self.assertEqual(
            [sample["filename"] for sample in manifest["validation"]],
            ["image_236.jpg", "image_300.jpg"],
        )
        self.assertFalse(manifest["train"][0]["blink"])
        self.assertTrue(manifest["train"][1]["blink"])
        self.assertEqual(
            manifest["summary"],
            {
                "train": {"total": 2, "pupil": 1, "blink": 1},
                "validation": {"total": 2, "pupil": 1, "blink": 1},
            },
        )

    def test_split_rejects_an_unparseable_image_filename(self):
        entries = [{"filename": "eye.jpg", "regions": []}]

        with self.assertRaisesRegex(ValueError, "image number"):
            build_split(entries, validation_start=236)


if __name__ == "__main__":
    unittest.main()
