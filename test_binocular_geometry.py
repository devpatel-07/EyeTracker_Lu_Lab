import math
import unittest

import numpy as np

from binocular_geometry import rotate_about_y, to_shared_coordinates


class BinocularGeometryTests(unittest.TestCase):
    def test_converts_camera_y_down_to_shared_y_up(self):
        result = to_shared_coordinates(
            "left",
            64.0,
            (0.0, 4.0, 35.0),
            (0.0, 3.0, 24.0),
        )

        self.assertEqual(result.shared_pupil_center_mm, (-32.0, 1.0, -11.0))
        self.assertGreater(result.shared_gaze_direction[1], 0.0)
        self.assertEqual(result.shared_camera_lens_mm, (-32.0, 4.0, -35.0))

    def test_anchors_eye_centers_and_converts_pupil_displacement(self):
        left = to_shared_coordinates(
            "left",
            64.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
        )
        right = to_shared_coordinates(
            "right",
            64.0,
            (11.0, 5.0, 36.0),
            (8.0, 7.0, 25.0),
        )

        self.assertEqual(left.shared_eye_center_mm, (-32.0, 0.0, 0.0))
        self.assertEqual(right.shared_eye_center_mm, (32.0, 0.0, 0.0))
        self.assertEqual(
            left.shared_pupil_center_mm,
            (-30.0, 1.0, -11.0),
        )
        self.assertEqual(
            right.shared_pupil_center_mm,
            (29.0, -2.0, -11.0),
        )
        self.assertAlmostEqual(
            math.dist(
                left.shared_eye_center_mm,
                right.shared_eye_center_mm,
            ),
            64.0,
        )
        local_displacement = np.subtract(
            left.local_pupil_center_mm,
            left.local_eye_center_mm,
        )
        expected_shared_displacement = local_displacement * (1.0, -1.0, 1.0)
        np.testing.assert_allclose(
            np.subtract(
                left.shared_pupil_center_mm,
                left.shared_eye_center_mm,
            ),
            expected_shared_displacement,
        )

    def test_rejects_invalid_side(self):
        with self.assertRaisesRegex(ValueError, "side"):
            to_shared_coordinates(
                "center",
                64.0,
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            )

    def test_rejects_invalid_separation(self):
        with self.assertRaisesRegex(ValueError, "separation"):
            to_shared_coordinates(
                "left",
                0.0,
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            )

    def test_rejects_nonfinite_coordinate(self):
        with self.assertRaisesRegex(ValueError, "finite 3D"):
            to_shared_coordinates(
                "left",
                64.0,
                (math.nan, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            )

    def test_rotates_left_geometry_and_recovers_lens_position(self):
        yaw_degrees = 35.0
        eye_local = (-10.0, 4.0, 35.0)
        pupil_local = (-8.0, 3.0, 24.0)
        result = to_shared_coordinates(
            "left",
            64.0,
            eye_local,
            pupil_local,
            camera_yaw_degrees=yaw_degrees,
        )

        yaw = math.radians(yaw_degrees)
        local_displacement = np.subtract(pupil_local, eye_local) * (
            1.0,
            -1.0,
            1.0,
        )
        expected_displacement = np.array(
            (
                math.cos(yaw) * local_displacement[0]
                + math.sin(yaw) * local_displacement[2],
                local_displacement[1],
                -math.sin(yaw) * local_displacement[0]
                + math.cos(yaw) * local_displacement[2],
            )
        )
        expected_pupil = np.array((-32.0, 0.0, 0.0)) + expected_displacement
        shared_axis_eye = np.asarray(eye_local) * (1.0, -1.0, 1.0)
        rotated_eye = np.asarray(rotate_about_y(shared_axis_eye, yaw_degrees))
        expected_lens = np.array((-32.0, 0.0, 0.0)) - rotated_eye

        np.testing.assert_allclose(
            result.shared_pupil_center_mm,
            expected_pupil,
        )
        np.testing.assert_allclose(
            result.shared_camera_lens_mm,
            expected_lens,
        )
        np.testing.assert_allclose(
            np.add(
                result.shared_camera_lens_mm,
                rotated_eye,
            ),
            result.shared_eye_center_mm,
        )
        self.assertAlmostEqual(
            np.linalg.norm(result.shared_gaze_direction),
            1.0,
        )
        np.testing.assert_allclose(
            result.shared_gaze_direction,
            expected_displacement / np.linalg.norm(expected_displacement),
        )

    def test_right_camera_uses_opposite_yaw(self):
        local_vector = (2.0, -1.0, -11.0)
        left_rotation = rotate_about_y(local_vector, 35.0)
        right_rotation = rotate_about_y(local_vector, -35.0)

        self.assertNotEqual(left_rotation, right_rotation)
        self.assertAlmostEqual(left_rotation[1], right_rotation[1])
        self.assertAlmostEqual(
            np.linalg.norm(left_rotation),
            np.linalg.norm(right_rotation),
        )

    def test_zero_yaw_still_converts_vertical_axis(self):
        result = to_shared_coordinates(
            "left",
            64.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
            camera_yaw_degrees=0.0,
        )

        self.assertEqual(
            result.shared_pupil_center_mm,
            (-30.0, 1.0, -11.0),
        )

    def test_rejects_invalid_yaw(self):
        with self.assertRaisesRegex(ValueError, "yaw"):
            to_shared_coordinates(
                "left",
                64.0,
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
                camera_yaw_degrees=math.nan,
            )

    def test_rejects_zero_length_gaze_vector(self):
        with self.assertRaisesRegex(ValueError, "gaze"):
            to_shared_coordinates(
                "left",
                64.0,
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                camera_yaw_degrees=35.0,
            )


if __name__ == "__main__":
    unittest.main()
