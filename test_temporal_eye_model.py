import unittest

import numpy as np

try:
    from temporal_eye_model import (
        PYE3D_REFERENCE_EYE_RADIUS_MM,
        TemporalEyeModel,
    )
except ImportError as exc:
    TemporalEyeModel = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def update_and_detect(
        self,
        pupil_datum,
        frame,
        apply_refraction_correction=True,
    ):
        self.calls.append(
            (pupil_datum, frame.copy(), apply_refraction_correction)
        )
        return self.result


class TemporalEyeModelTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"temporal_eye_model is not implemented: {IMPORT_ERROR}")

        self.detector = FakeDetector(
            {
                "sphere": {"center": (1.0, 2.0, 30.0), "radius": 10.4},
                "circle_3d": {
                    "center": (0.5, 1.5, 20.0),
                    "normal": (0.0, 0.0, -1.0),
                    "radius": 1.6,
                },
                "diameter_3d": 3.2,
                "confidence": 0.9,
                "model_confidence": 1.0,
            }
        )
        self.factory_arguments = None

        def factory(focal_length, resolution, min_confidence):
            self.factory_arguments = (
                focal_length,
                resolution,
                min_confidence,
            )
            return self.detector

        self.model = TemporalEyeModel(
            camera_matrix=np.array(
                [
                    [1000.0, 0.0, 530.0],
                    [0.0, 1002.0, 950.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            resolution=(1080, 1920),
            min_confidence=0.60,
            detector_factory=factory,
        )
        self.grayscale = np.zeros((1920, 1080), dtype=np.uint8)

    def test_recenters_and_canonicalizes_observation(self):
        result = self.model.update(
            ellipse=((530.0, 950.0), (80.0, 50.0), 20.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertEqual(self.factory_arguments, (1001.0, (1080, 1920), 0.60))
        self.assertEqual(len(self.detector.calls), 1)
        datum, passed_frame, apply_refraction = self.detector.calls[0]
        self.assertEqual(datum["ellipse"]["center"], (540.0, 960.0))
        self.assertEqual(datum["ellipse"]["axes"], (50.0, 80.0))
        self.assertEqual(datum["ellipse"]["angle"], 110.0)
        self.assertEqual(datum["diameter"], 80.0)
        self.assertEqual(datum["location"], (540.0, 960.0))
        self.assertEqual(datum["confidence"], 0.9)
        self.assertEqual(datum["timestamp"], 1.0)
        self.assertEqual(datum["method"], "custom-ml")
        self.assertIs(passed_frame.dtype, self.grayscale.dtype)
        self.assertFalse(apply_refraction)
        self.assertTrue(result.ready)

    def test_extracts_non_refractive_geometry(self):
        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 110.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        scale = 12.0 / PYE3D_REFERENCE_EYE_RADIUS_MM
        np.testing.assert_allclose(
            result.eye_center_mm,
            np.asarray((1.0, 2.0, 30.0)) * scale,
        )
        np.testing.assert_allclose(
            result.pupil_center_mm,
            np.asarray((0.5, 1.5, 20.0)) * scale,
        )
        self.assertAlmostEqual(result.raw_diameter_mm, 3.2 * scale)
        self.assertIsNone(result.corrected_diameter_mm)
        self.assertEqual(result.confidence, 0.9)
        self.assertGreaterEqual(result.update_time_ms, 0.0)
        self.assertIs(result.raw_result, self.detector.result)

    def test_rejects_invalid_assumed_eye_radius(self):
        with self.assertRaisesRegex(ValueError, "eye_radius_mm"):
            TemporalEyeModel(
                camera_matrix=np.eye(3),
                resolution=(1080, 1920),
                eye_radius_mm=0.0,
                detector_factory=lambda *args: self.detector,
            )

    def test_rejects_low_confidence_without_updating_detector(self):
        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
            confidence=0.59,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertFalse(result.ready)
        self.assertEqual(len(self.detector.calls), 0)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "axes"):
            self.model.update(
                ellipse=((530.0, 950.0), (0.0, 80.0), 20.0),
                confidence=0.9,
                timestamp=1.0,
                grayscale=self.grayscale,
            )

        with self.assertRaisesRegex(ValueError, "frame size"):
            self.model.update(
                ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
                confidence=0.9,
                timestamp=1.0,
                grayscale=np.zeros((100, 100), dtype=np.uint8),
            )

    def test_rejects_non_monotonic_accepted_timestamp(self):
        observation = {
            "ellipse": ((530.0, 950.0), (50.0, 80.0), 20.0),
            "confidence": 0.9,
            "grayscale": self.grayscale,
        }
        self.model.update(timestamp=1.0, **observation)

        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.model.update(timestamp=1.0, **observation)

    def test_incomplete_model_output_is_not_ready(self):
        self.detector.result = {
            "sphere": {"center": (1.0, 2.0, 30.0)},
            "circle_3d": {"center": (0.0, 0.0, 0.0), "radius": 0.0},
            "diameter_3d": 0.0,
            "confidence": 0.9,
            "model_confidence": 0.1,
        }

        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertFalse(result.ready)
        self.assertIsNone(result.pupil_center_mm)
        self.assertIsNone(result.raw_diameter_mm)

    def test_off_axis_camera_position_does_not_block_valid_geometry(self):
        self.detector.result["sphere"]["center"] = (1.0, -14.0, 30.0)
        self.detector.result["model_confidence"] = 0.1

        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertTrue(result.ready)

    def test_implausible_depth_is_not_ready(self):
        self.detector.result["sphere"]["center"] = (1.0, 2.0, 100.0)

        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertFalse(result.ready)

    def test_implausible_scaled_pupil_diameter_is_not_ready(self):
        self.detector.result["circle_3d"]["radius"] = 5.0
        self.detector.result["diameter_3d"] = 10.0

        result = self.model.update(
            ellipse=((530.0, 950.0), (50.0, 80.0), 20.0),
            confidence=0.9,
            timestamp=1.0,
            grayscale=self.grayscale,
        )

        self.assertFalse(result.ready)


if __name__ == "__main__":
    unittest.main()
