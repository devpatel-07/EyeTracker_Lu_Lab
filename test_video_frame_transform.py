import math
import unittest
from dataclasses import FrozenInstanceError

import cv2
import numpy as np

from video_frame_transform import FrameTransform, TransformedVideoCapture


class FakeCapture:
    def __init__(self, frames, width=4, height=3):
        self.frames = list(frames)
        self.width = width
        self.height = height
        self.get_calls = []
        self.is_opened_calls = 0
        self.release_calls = 0

    def read(self):
        return self.frames.pop(0)

    def get(self, property_id):
        self.get_calls.append(property_id)
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if property_id == cv2.CAP_PROP_FPS:
            return 29.97
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return 123.5
        return -1.0

    def isOpened(self):
        self.is_opened_calls += 1
        return True

    def release(self):
        self.release_calls += 1


class FrameTransformTests(unittest.TestCase):
    def test_rejects_invalid_settings(self):
        with self.assertRaisesRegex(ValueError, "rotation"):
            FrameTransform(rotation="sideways")
        with self.assertRaisesRegex(ValueError, "rotation"):
            FrameTransform(rotation=[])
        with self.assertRaisesRegex(ValueError, "flip_horizontal"):
            FrameTransform(flip_horizontal=1)
        with self.assertRaisesRegex(ValueError, "flip_vertical"):
            FrameTransform(flip_vertical=None)

    def test_settings_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            FrameTransform().rotation = "180"

    def test_rotation_output_sizes(self):
        self.assertEqual(FrameTransform().output_size((640, 480)), (640, 480))
        self.assertEqual(
            FrameTransform("clockwise").output_size((640, 480)),
            (480, 640),
        )
        self.assertEqual(
            FrameTransform("counterclockwise").output_size((640, 480)),
            (480, 640),
        )
        self.assertEqual(
            FrameTransform("180").output_size((640, 480)),
            (640, 480),
        )

    def test_applies_rotation_then_both_flips(self):
        frame = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        transform = FrameTransform(
            rotation="clockwise",
            flip_horizontal=True,
            flip_vertical=True,
        )
        expected = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        expected = cv2.flip(expected, 1)
        expected = cv2.flip(expected, 0)

        np.testing.assert_array_equal(transform.apply_frame(frame), expected)

    def test_point_round_trip_for_every_combination(self):
        points = np.asarray(((0.0, 0.0), (12.5, 7.25), (639.0, 479.0)))
        for rotation in ("none", "clockwise", "counterclockwise", "180"):
            for flip_horizontal in (False, True):
                for flip_vertical in (False, True):
                    with self.subTest(
                        rotation=rotation,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                    ):
                        transform = FrameTransform(
                            rotation,
                            flip_horizontal,
                            flip_vertical,
                        )
                        transformed = transform.forward_points(
                            points,
                            (640, 480),
                        )
                        restored = transform.inverse_points(
                            transformed,
                            (640, 480),
                        )
                        np.testing.assert_allclose(restored, points)

    def test_camera_axis_round_trip_preserves_vector_length(self):
        transform = FrameTransform(
            rotation="clockwise",
            flip_horizontal=True,
            flip_vertical=False,
        )
        source = (2.0, -3.0, 10.0)
        runtime = transform.to_runtime_camera_axes(source)
        restored = transform.to_source_camera_axes(runtime)

        np.testing.assert_allclose(restored, source)
        self.assertAlmostEqual(np.linalg.norm(restored), np.linalg.norm(runtime))

    def test_camera_matrix_transforms_intrinsics_and_principal_point(self):
        matrix = np.array(
            [[800.0, 0.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]]
        )
        rotated = FrameTransform("clockwise").transform_camera_matrix(
            matrix, (640, 480)
        )
        np.testing.assert_allclose(
            rotated,
            [[900.0, 0.0, 480 - 1 - 200], [0.0, 800.0, 300], [0.0, 0.0, 1.0]],
        )

        flipped = FrameTransform(
            "clockwise",
            flip_horizontal=True,
            flip_vertical=True,
        ).transform_camera_matrix(matrix, (640, 480))
        self.assertEqual(flipped[0, 2], 200.0)
        self.assertEqual(flipped[1, 2], 339.0)
        self.assertGreater(flipped[0, 0], 0.0)
        self.assertGreater(flipped[1, 1], 0.0)

    def test_point_transform_requires_n_by_two_points_and_returns_float64(self):
        transform = FrameTransform()
        for points in (np.zeros(2), np.zeros((2, 3)), np.zeros((1, 2, 1))):
            with self.subTest(shape=points.shape):
                with self.assertRaisesRegex(ValueError, r"points must have shape \(N, 2\)"):
                    transform.forward_points(points, (4, 3))

        transformed = transform.forward_points(
            np.asarray([[1, 2]], dtype=np.float32), (4, 3)
        )
        self.assertEqual(transformed.dtype, np.float64)

    def test_camera_matrix_rejects_non_pinhole_values(self):
        transform = FrameTransform()
        invalid_matrices = (
            np.eye(2),
            np.array([[800.0, 1.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]]),
            np.array([[np.nan, 0.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]]),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "camera matrix"):
                    transform.transform_camera_matrix(matrix, (640, 480))


class TransformedVideoCaptureTests(unittest.TestCase):
    def test_transforms_frames_and_forwards_capture_behavior(self):
        frame = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        failed_read = (False, None)
        capture = FakeCapture([(True, frame), failed_read])
        transform = FrameTransform("clockwise", flip_horizontal=True)
        adapted = TransformedVideoCapture(capture, transform)

        self.assertEqual(adapted.get(cv2.CAP_PROP_FRAME_WIDTH), 3.0)
        self.assertEqual(adapted.get(cv2.CAP_PROP_FRAME_HEIGHT), 4.0)
        self.assertEqual(adapted.get(cv2.CAP_PROP_FPS), 29.97)
        self.assertEqual(adapted.get(cv2.CAP_PROP_POS_MSEC), 123.5)

        successful, transformed = adapted.read()
        self.assertTrue(successful)
        np.testing.assert_array_equal(transformed, transform.apply_frame(frame))
        self.assertEqual(adapted.read(), failed_read)
        self.assertTrue(adapted.isOpened())
        self.assertEqual(capture.is_opened_calls, 1)

        adapted.release()
        adapted.release()
        self.assertEqual(capture.release_calls, 1)

    def test_does_not_transform_successful_none_frames(self):
        capture = FakeCapture([(True, None)])
        adapted = TransformedVideoCapture(capture, FrameTransform("180"))

        self.assertEqual(adapted.read(), (True, None))


if __name__ == "__main__":
    unittest.main()
