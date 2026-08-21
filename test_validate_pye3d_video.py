import unittest
import math

from temporal_eye_model import EyeModelResult

try:
    from validate_pye3d_video import summarize_results
except ImportError as exc:
    summarize_results = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def make_result(
    ready,
    diameter,
    update_time_ms,
    eye_center=(1.0, 2.0, 30.0),
    pupil_center=(0.5, 1.5, 20.0),
    confidence=0.9,
    raw_result=None,
):
    return EyeModelResult(
        ready=ready,
        eye_center_mm=eye_center if ready else None,
        pupil_center_mm=pupil_center if ready else None,
        raw_diameter_mm=diameter if ready else None,
        corrected_diameter_mm=None,
        confidence=confidence,
        update_time_ms=update_time_ms,
        raw_result=raw_result,
    )


class ValidatePye3DVideoTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"validate_pye3d_video is not implemented: {IMPORT_ERROR}")

    def test_summary_reports_geometry_and_timing_statistics(self):
        results = [
            make_result(False, None, 1.0, confidence=0.4),
            make_result(
                True,
                3.0,
                2.0,
                raw_result={
                    "model_confidence": 1.0,
                    "diameter_3d": 3.0,
                    "sphere": {"center": (1.0, 2.0, 30.0)},
                    "phi": -math.pi / 2,
                    "theta": math.pi / 2,
                },
            ),
            make_result(
                True,
                4.0,
                3.0,
                raw_result={
                    "model_confidence": 1.0,
                    "diameter_3d": 4.0,
                    "sphere": {"center": (1.0, 2.0, 31.0)},
                    "phi": -math.pi / 2,
                    "theta": math.pi / 2,
                },
            ),
            make_result(
                True,
                5.0,
                4.0,
                eye_center=(float("nan"), 0.0, 30.0),
                raw_result={
                    "model_confidence": 1.0,
                    "diameter_3d": 5.0,
                    "sphere": {"center": (1.0, 2.0, 32.0)},
                    "phi": -math.pi / 2,
                    "theta": math.pi / 2,
                },
            ),
            make_result(
                False,
                None,
                5.0,
                raw_result={
                    "model_confidence": 0.1,
                    "diameter_3d": 0.5,
                    "sphere": {"center": (20.0, 12.0, 80.0)},
                    "phi": -math.pi / 2,
                    "theta": math.pi / 2,
                },
            ),
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["accepted_detections"], 5)
        self.assertEqual(summary["ready_outputs"], 3)
        self.assertEqual(summary["finite_coordinate_failures"], 1)
        self.assertEqual(summary["pye3d_updates"], 4)
        self.assertEqual(summary["pre_pye3d_rejections"], 1)
        self.assertEqual(summary["model_out_of_range_outputs"], 1)
        self.assertEqual(
            summary["pye3d_out_of_range_reasons"],
            {
                "eye_center_x": 1,
                "eye_center_y": 1,
                "eye_center_z": 1,
                "pupil_diameter": 1,
                "gaze_angles": 0,
            },
        )
        self.assertEqual(summary["diameter_mm"]["minimum"], 3.0)
        self.assertEqual(summary["diameter_mm"]["median"], 4.0)
        self.assertEqual(summary["diameter_mm"]["maximum"], 5.0)
        self.assertEqual(summary["eye_center_z_mm"]["minimum"], 30.0)
        self.assertEqual(summary["eye_center_z_mm"]["median"], 30.0)
        self.assertEqual(summary["eye_center_z_mm"]["maximum"], 30.0)
        self.assertEqual(summary["pupil_center_z_mm"]["median"], 20.0)
        self.assertEqual(summary["update_time_ms"]["median"], 3.0)
        self.assertAlmostEqual(summary["update_time_ms"]["p95"], 4.8)
        self.assertEqual(summary["raw_model_diameter_mm"]["minimum"], 0.5)
        self.assertEqual(summary["raw_eye_center_z_mm"]["maximum"], 80.0)

    def test_empty_summary_has_no_statistics(self):
        summary = summarize_results([])

        self.assertEqual(summary["accepted_detections"], 0)
        self.assertEqual(summary["ready_outputs"], 0)
        self.assertEqual(summary["finite_coordinate_failures"], 0)
        self.assertEqual(summary["pye3d_updates"], 0)
        self.assertEqual(summary["pre_pye3d_rejections"], 0)
        self.assertEqual(summary["model_out_of_range_outputs"], 0)
        self.assertEqual(
            summary["pye3d_out_of_range_reasons"],
            {
                "eye_center_x": 0,
                "eye_center_y": 0,
                "eye_center_z": 0,
                "pupil_diameter": 0,
                "gaze_angles": 0,
            },
        )
        self.assertIsNone(summary["diameter_mm"])
        self.assertIsNone(summary["eye_center_z_mm"])
        self.assertIsNone(summary["pupil_center_z_mm"])
        self.assertIsNone(summary["update_time_ms"])


if __name__ == "__main__":
    unittest.main()
