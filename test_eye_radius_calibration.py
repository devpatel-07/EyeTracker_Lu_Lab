import unittest

try:
    from eye_radius_calibration import EyeRadiusCalibrator
except ImportError as exc:
    EyeRadiusCalibrator = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class EyeRadiusCalibratorTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"eye_radius_calibration is not implemented: {IMPORT_ERROR}")

    def test_invalid_and_out_of_bounds_samples_are_rejected(self):
        calibrator = EyeRadiusCalibrator(
            target_samples=2,
            min_radius_px=10,
            max_radius_px=100,
        )

        self.assertFalse(calibrator.add(None))
        self.assertFalse(calibrator.add(float("nan")))
        self.assertFalse(calibrator.add(9))
        self.assertFalse(calibrator.add(101))
        self.assertTrue(calibrator.add(20))

        self.assertEqual(calibrator.sample_count, 1)
        self.assertEqual(calibrator.progress, 0.5)
        self.assertFalse(calibrator.ready)

    def test_calibration_uses_requested_percentile_and_freezes(self):
        calibrator = EyeRadiusCalibrator(
            target_samples=5,
            percentile=95,
            min_radius_px=1,
            max_radius_px=1000,
        )
        for value in (10, 20, 30, 40, 50):
            self.assertTrue(calibrator.add(value))

        self.assertTrue(calibrator.ready)
        self.assertAlmostEqual(calibrator.radius_px, 48.0)
        self.assertEqual(calibrator.progress, 1.0)

        self.assertFalse(calibrator.add(900))
        self.assertAlmostEqual(calibrator.radius_px, 48.0)

    def test_95th_percentile_ignores_a_small_number_of_high_samples(self):
        calibrator = EyeRadiusCalibrator(
            target_samples=100,
            percentile=95,
            min_radius_px=1,
            max_radius_px=300,
        )
        for value in [50] * 96 + [200] * 4:
            calibrator.add(value)

        self.assertAlmostEqual(calibrator.radius_px, 50.0)


if __name__ == "__main__":
    unittest.main()
