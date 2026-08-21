import unittest

import numpy as np

from video_frame_transform import FrameTransform

try:
    from camera_geometry import (
        distort_points,
        rotated_camera_matrix,
        undistort_ellipse,
        undistort_points,
    )
except ImportError as exc:
    distort_points = None
    rotated_camera_matrix = None
    undistort_ellipse = None
    undistort_points = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"camera_geometry is not implemented: {IMPORT_ERROR}")

    def test_clockwise_camera_matrix_swaps_axes_and_rotates_center(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 20.0], [0.0, 200.0, 30.0], [0.0, 0.0, 1.0]]
        )

        rotated = rotated_camera_matrix(
            camera_matrix, raw_image_size=(100, 60), rotation="clockwise"
        )

        expected = np.array(
            [[200.0, 0.0, 29.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(rotated, expected)

    def test_zero_distortion_preserves_clockwise_points(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 49.5], [0.0, 100.0, 29.5], [0.0, 0.0, 1.0]]
        )
        points = np.array([[10.0, 20.0], [30.0, 70.0]], dtype=np.float32)

        corrected = undistort_points(
            points,
            camera_matrix,
            np.zeros((1, 5)),
            raw_image_size=(100, 60),
            rotation="clockwise",
        )

        np.testing.assert_allclose(corrected, points, atol=1e-4)

    def test_saved_calibration_corrects_point_in_clockwise_eye_region(self):
        camera_matrix = np.array(
            [
                [1417.664857809249, 0.0, 939.8180262747536],
                [0.0, 1414.1417640929117, 539.0781008125922],
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.array(
            [[-0.42774571, 0.28128514, 0.00089517, 0.00347835, -0.1470389]]
        )

        corrected = undistort_points(
            np.array([[540.0, 300.0]], dtype=np.float32),
            camera_matrix,
            distortion,
            raw_image_size=(1920, 1080),
            rotation="clockwise",
        )

        np.testing.assert_allclose(corrected[0], [540.3574, 231.4702], atol=0.05)

    def test_distort_and_undistort_points_round_trip(self):
        camera_matrix = np.array(
            [
                [1417.664857809249, 0.0, 939.8180262747536],
                [0.0, 1414.1417640929117, 539.0781008125922],
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.array(
            [[-0.42774571, 0.28128514, 0.00089517, 0.00347835, -0.1470389]]
        )
        points = np.array([[270.0, 300.0], [540.0, 500.0], [800.0, 300.0]])

        corrected = undistort_points(
            points, camera_matrix, distortion, (1920, 1080), "clockwise"
        )
        restored = distort_points(
            corrected, camera_matrix, distortion, (1920, 1080), "clockwise"
        )

        np.testing.assert_allclose(restored, points, atol=0.05)

    def test_distort_and_undistort_round_trip_with_runtime_transforms(self):
        camera_matrix = np.array(
            [
                [1417.664857809249, 0.0, 939.8180262747536],
                [0.0, 1414.1417640929117, 539.0781008125922],
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.array(
            [[-0.42774571, 0.28128514, 0.00089517, 0.00347835, -0.1470389]]
        )
        source_size = (1920, 1080)
        source_points = np.array(
            [[600.0, 250.0], [900.0, 500.0], [1200.0, 750.0]]
        )

        for rotation in ("none", "clockwise", "counterclockwise", "180"):
            with self.subTest(rotation=rotation):
                transform = FrameTransform(
                    rotation=rotation,
                    flip_horizontal=True,
                    flip_vertical=True,
                )
                runtime_points = transform.forward_points(
                    source_points,
                    source_size,
                )

                distorted = distort_points(
                    runtime_points,
                    camera_matrix,
                    distortion,
                    source_size,
                    frame_transform=transform,
                )
                restored = undistort_points(
                    distorted,
                    camera_matrix,
                    distortion,
                    source_size,
                    frame_transform=transform,
                )

                np.testing.assert_allclose(restored, runtime_points, atol=0.05)

    def test_zero_distortion_ellipse_refit_preserves_geometry(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 49.5], [0.0, 100.0, 29.5], [0.0, 0.0, 1.0]]
        )
        ellipse = ((30.0, 70.0), (20.0, 40.0), 25.0)

        corrected = undistort_ellipse(
            ellipse,
            camera_matrix,
            np.zeros((1, 5)),
            raw_image_size=(100, 60),
            rotation="clockwise",
        )

        np.testing.assert_allclose(corrected[0], ellipse[0], atol=0.1)
        np.testing.assert_allclose(
            sorted(corrected[1]), sorted(ellipse[1]), atol=0.2
        )

    def test_undistort_ellipse_with_runtime_transform_preserves_center(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 49.5], [0.0, 100.0, 29.5], [0.0, 0.0, 1.0]]
        )
        source_size = (100, 60)
        source_ellipse = ((30.0, 20.0), (20.0, 30.0), 25.0)
        transform = FrameTransform(rotation="clockwise", flip_horizontal=True)
        runtime_center = transform.forward_points(
            np.array([source_ellipse[0]]),
            source_size,
        )[0]
        runtime_ellipse = (
            tuple(runtime_center),
            source_ellipse[1],
            source_ellipse[2],
        )

        corrected = undistort_ellipse(
            runtime_ellipse,
            camera_matrix,
            np.zeros((1, 5)),
            raw_image_size=source_size,
            frame_transform=transform,
        )

        self.assertTrue(np.all(np.isfinite(corrected[0])))
        self.assertTrue(np.all(np.isfinite(corrected[1])))
        np.testing.assert_allclose(corrected[0], runtime_center, atol=0.1)


if __name__ == "__main__":
    unittest.main()
