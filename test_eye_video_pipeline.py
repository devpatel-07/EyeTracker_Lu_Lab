import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from binocular_geometry import to_shared_coordinates
from eye_video_pipeline import (
    EyeCameraCalibration,
    EyeVideoPipeline,
    read_frame_pair,
    validate_synchronized_captures,
)
from temporal_eye_model import EyeModelResult
from video_frame_transform import FrameTransform


class FakeCapture:
    def __init__(
        self,
        frames=(),
        fps=30.0,
        width=60,
        height=80,
        timestamp_ms=1000.0,
        opened=True,
    ):
        self.frames = list(frames)
        self.fps = fps
        self.width = width
        self.height = height
        self.timestamp_ms = timestamp_ms
        self.opened = opened

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_POS_MSEC: self.timestamp_ms,
        }
        return values.get(property_id, 0.0)


class FakePupilDetector:
    def __init__(self):
        self.calls = []

    def detect_with_confidence(self, frame, rect, debug=False):
        self.calls.append((frame.copy(), dict(rect), debug))
        ellipse = ((30.0, 40.0), (12.0, 20.0), 10.0)
        return frame.copy(), ellipse, False, 0.9


class FakeTemporalModel:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or EyeModelResult(
            ready=True,
            eye_center_mm=(-10.0, 4.0, 35.0),
            pupil_center_mm=(-8.0, 3.0, 24.0),
            raw_diameter_mm=3.2,
            corrected_diameter_mm=None,
            confidence=0.9,
            update_time_ms=1.5,
            raw_result={},
        )

    def update(self, ellipse, confidence, timestamp, grayscale):
        self.calls.append((ellipse, confidence, timestamp, grayscale.copy()))
        return self.result


class FakeGraph:
    def __init__(self, **_kwargs):
        self.starts = []
        self.samples = []

    def start(self, timestamp):
        self.starts.append(timestamp)
        return True

    def add_sample(self, timestamp, diameter):
        self.samples.append((timestamp, diameter))
        return True

    def render(self):
        return np.zeros((20, 30, 3), dtype=np.uint8)


class EyeVideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.calibration_path = Path(self.temp_dir.name) / "camera.npz"
        np.savez(
            self.calibration_path,
            camera_matrix=np.array(
                [
                    [100.0, 0.0, 39.5],
                    [0.0, 101.0, 29.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
            dist_coeffs=np.zeros((1, 5)),
            image_size=np.array((80, 60)),
        )

    def test_loads_clockwise_calibration(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )

        self.assertEqual(calibration.raw_image_size, (80, 60))
        self.assertEqual(calibration.expected_frame_size, (60, 80))
        np.testing.assert_allclose(
            calibration.camera_matrix,
            np.array(
                [
                    [101.0, 0.0, 29.5],
                    [0.0, 100.0, 39.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_loads_calibration_with_runtime_transform(self):
        transform = FrameTransform(
            rotation="clockwise",
            flip_horizontal=True,
            flip_vertical=True,
        )

        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "none",
            transform,
        )

        self.assertEqual(calibration.source_frame_size, (80, 60))
        self.assertEqual(
            calibration.expected_frame_size,
            transform.output_size(calibration.source_frame_size),
        )
        np.testing.assert_allclose(
            calibration.camera_matrix,
            transform.transform_camera_matrix(
                calibration.raw_camera_matrix,
                calibration.source_frame_size,
            ),
        )
        self.assertIs(calibration.frame_transform, transform)

    def test_loads_calibration_with_default_identity_transform(self):
        calibration = EyeCameraCalibration.load(self.calibration_path, "none")

        self.assertEqual(calibration.source_frame_size, (80, 60))
        self.assertEqual(calibration.expected_frame_size, (80, 60))
        np.testing.assert_allclose(
            calibration.camera_matrix,
            calibration.raw_camera_matrix,
        )
        self.assertEqual(calibration.frame_transform, FrameTransform())

    def test_capture_validation_rejects_mismatched_frame_rates(self):
        left = FakeCapture(fps=30.0)
        right = FakeCapture(fps=25.0)

        with self.assertRaisesRegex(ValueError, "frame rates"):
            validate_synchronized_captures(left, right)

    def test_capture_validation_rejects_closed_capture(self):
        with self.assertRaisesRegex(ValueError, "left video"):
            validate_synchronized_captures(
                FakeCapture(opened=False),
                FakeCapture(),
            )

    def test_read_frame_pair_stops_when_either_stream_ends(self):
        frame = np.zeros((80, 60, 3), dtype=np.uint8)
        left = FakeCapture(frames=[frame, frame])
        right = FakeCapture(frames=[])

        self.assertIsNone(read_frame_pair(left, right))

    def test_pipeline_transforms_ready_model_result(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )
        capture = FakeCapture()
        pupil_detector = FakePupilDetector()
        created_models = []

        def model_factory(**_kwargs):
            model = FakeTemporalModel()
            created_models.append(model)
            return model

        pipeline = EyeVideoPipeline(
            side="left",
            capture=capture,
            calibration=calibration,
            pupil_detector=pupil_detector,
            eye_center_separation_mm=64.0,
            temporal_model_factory=model_factory,
            graph_factory=FakeGraph,
        )
        frame = np.zeros((80, 60, 3), dtype=np.uint8)

        output = pipeline.process_frame(frame)

        self.assertTrue(output.model_result.ready)
        self.assertEqual(
            output.coordinates.shared_eye_center_mm,
            (-32.0, 0.0, 0.0),
        )
        self.assertEqual(
            output.coordinates.shared_pupil_center_mm,
            (-30.0, 1.0, -11.0),
        )
        self.assertEqual(output.pupil_diameter_mm, 3.2)
        self.assertEqual(len(created_models[0].calls), 1)
        self.assertEqual(pipeline.graph.samples, [(1.0, 3.2)])

    def test_pipeline_preserves_shared_coordinates_across_runtime_camera_axes(self):
        source_eye = (-10.0, 4.0, 35.0)
        source_pupil = (-8.0, 3.0, 24.0)
        transform = FrameTransform(
            rotation="clockwise",
            flip_horizontal=True,
        )
        runtime_eye = transform.to_runtime_camera_axes(source_eye)
        runtime_pupil = transform.to_runtime_camera_axes(source_pupil)
        identity_result = EyeModelResult(
            ready=True,
            eye_center_mm=source_eye,
            pupil_center_mm=source_pupil,
            raw_diameter_mm=3.2,
            corrected_diameter_mm=None,
            confidence=0.9,
            update_time_ms=1.5,
            raw_result={},
        )
        transformed_result = EyeModelResult(
            ready=True,
            eye_center_mm=runtime_eye,
            pupil_center_mm=runtime_pupil,
            raw_diameter_mm=3.2,
            corrected_diameter_mm=None,
            confidence=0.9,
            update_time_ms=1.5,
            raw_result={},
        )
        identity_calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "none",
        )
        transformed_calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "none",
            transform,
        )

        identity_pipeline = EyeVideoPipeline(
            side="left",
            capture=FakeCapture(width=80, height=60),
            calibration=identity_calibration,
            pupil_detector=FakePupilDetector(),
            eye_center_separation_mm=64.0,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(
                identity_result
            ),
            graph_factory=FakeGraph,
        )
        transformed_pipeline = EyeVideoPipeline(
            side="left",
            capture=FakeCapture(width=60, height=80),
            calibration=transformed_calibration,
            pupil_detector=FakePupilDetector(),
            eye_center_separation_mm=64.0,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(
                transformed_result
            ),
            graph_factory=FakeGraph,
        )

        identity_output = identity_pipeline.process_frame(
            np.zeros((60, 80, 3), dtype=np.uint8)
        )
        transformed_output = transformed_pipeline.process_frame(
            np.zeros((80, 60, 3), dtype=np.uint8)
        )

        np.testing.assert_allclose(
            identity_output.coordinates.shared_eye_center_mm,
            transformed_output.coordinates.shared_eye_center_mm,
        )
        np.testing.assert_allclose(
            identity_output.coordinates.shared_pupil_center_mm,
            transformed_output.coordinates.shared_pupil_center_mm,
        )
        np.testing.assert_allclose(
            identity_output.coordinates.shared_camera_lens_mm,
            transformed_output.coordinates.shared_camera_lens_mm,
        )
        np.testing.assert_allclose(
            identity_output.coordinates.shared_gaze_direction,
            transformed_output.coordinates.shared_gaze_direction,
        )
        self.assertEqual(
            identity_output.pupil_diameter_mm,
            transformed_output.pupil_diameter_mm,
        )
        self.assertEqual(transformed_output.model_result.eye_center_mm, runtime_eye)
        self.assertEqual(
            transformed_output.model_result.pupil_center_mm,
            runtime_pupil,
        )

    def test_pipeline_passes_runtime_transform_to_ellipse_undistortion(self):
        transform = FrameTransform(flip_horizontal=True)
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "none",
            transform,
        )
        pipeline = EyeVideoPipeline(
            side="left",
            capture=FakeCapture(width=80, height=60),
            calibration=calibration,
            pupil_detector=FakePupilDetector(),
            eye_center_separation_mm=64.0,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
            graph_factory=FakeGraph,
        )
        ellipse = ((30.0, 40.0), (12.0, 20.0), 10.0)

        with patch(
            "eye_video_pipeline.undistort_ellipse",
            return_value=ellipse,
        ) as undistort:
            pipeline.process_frame(np.zeros((60, 80, 3), dtype=np.uint8))

        self.assertIs(undistort.call_args.kwargs["frame_transform"], transform)

    def test_two_pipelines_share_detector_but_not_temporal_models(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )
        pupil_detector = FakePupilDetector()
        created_models = []

        def model_factory(**_kwargs):
            model = FakeTemporalModel()
            created_models.append(model)
            return model

        left = EyeVideoPipeline(
            "left",
            FakeCapture(),
            calibration,
            pupil_detector,
            64.0,
            temporal_model_factory=model_factory,
            graph_factory=FakeGraph,
        )
        right = EyeVideoPipeline(
            "right",
            FakeCapture(),
            calibration,
            pupil_detector,
            64.0,
            temporal_model_factory=model_factory,
            graph_factory=FakeGraph,
        )

        self.assertIs(left.pupil_detector, right.pupil_detector)
        self.assertIsNot(left.temporal_model, right.temporal_model)
        self.assertEqual(len(created_models), 2)

    def test_pipeline_rejects_invalid_eye_separation_at_startup(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )

        with self.assertRaisesRegex(ValueError, "separation"):
            EyeVideoPipeline(
                side="left",
                capture=FakeCapture(),
                calibration=calibration,
                pupil_detector=FakePupilDetector(),
                eye_center_separation_mm=0.0,
                temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
                graph_factory=FakeGraph,
            )

    def test_pipeline_applies_configured_camera_yaw(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )
        pupil_detector = FakePupilDetector()

        left = EyeVideoPipeline(
            side="left",
            capture=FakeCapture(),
            calibration=calibration,
            pupil_detector=pupil_detector,
            eye_center_separation_mm=64.0,
            camera_yaw_degrees=35.0,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
            graph_factory=FakeGraph,
        )
        right = EyeVideoPipeline(
            side="right",
            capture=FakeCapture(),
            calibration=calibration,
            pupil_detector=pupil_detector,
            eye_center_separation_mm=64.0,
            camera_yaw_degrees=-35.0,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
            graph_factory=FakeGraph,
        )
        frame = np.zeros((80, 60, 3), dtype=np.uint8)

        left_output = left.process_frame(frame)
        right_output = right.process_frame(frame)

        expected_left = to_shared_coordinates(
            "left",
            64.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
            camera_yaw_degrees=35.0,
        )
        expected_right = to_shared_coordinates(
            "right",
            64.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
            camera_yaw_degrees=-35.0,
        )
        self.assertEqual(left_output.coordinates, expected_left)
        self.assertEqual(right_output.coordinates, expected_right)

    def test_pipeline_rejects_nonfinite_camera_yaw_at_startup(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )

        with self.assertRaisesRegex(ValueError, "yaw"):
            EyeVideoPipeline(
                side="left",
                capture=FakeCapture(),
                calibration=calibration,
                pupil_detector=FakePupilDetector(),
                eye_center_separation_mm=64.0,
                camera_yaw_degrees=float("nan"),
                temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
                graph_factory=FakeGraph,
            )

    def test_pipeline_copies_and_forwards_explicit_rect(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )
        pupil_detector = FakePupilDetector()
        configured_rect = {"x": 5, "y": 6, "w": 40, "h": 50}
        pipeline = EyeVideoPipeline(
            side="left",
            capture=FakeCapture(),
            calibration=calibration,
            pupil_detector=pupil_detector,
            eye_center_separation_mm=64.0,
            rect=configured_rect,
            temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
            graph_factory=FakeGraph,
        )
        configured_rect["x"] = 20

        pipeline.process_frame(np.zeros((80, 60, 3), dtype=np.uint8))

        self.assertEqual(
            pipeline.rect,
            {"x": 5, "y": 6, "w": 40, "h": 50},
        )
        self.assertEqual(
            pupil_detector.calls[0][1],
            {"x": 5, "y": 6, "w": 40, "h": 50},
        )

    def test_pipeline_rejects_invalid_explicit_rect(self):
        calibration = EyeCameraCalibration.load(
            self.calibration_path,
            "clockwise",
        )
        invalid_rectangles = (
            {"x": -1, "y": 0, "w": 20, "h": 20},
            {"x": 0, "y": 0, "w": 0, "h": 20},
            {"x": 0, "y": 0, "w": 20},
            {"x": 50, "y": 0, "w": 20, "h": 20},
            {"x": 0, "y": 70, "w": 20, "h": 20},
        )

        for rect in invalid_rectangles:
            with self.subTest(rect=rect), self.assertRaisesRegex(
                ValueError,
                "rect",
            ):
                EyeVideoPipeline(
                    side="left",
                    capture=FakeCapture(),
                    calibration=calibration,
                    pupil_detector=FakePupilDetector(),
                    eye_center_separation_mm=64.0,
                    rect=rect,
                    temporal_model_factory=lambda **_kwargs: FakeTemporalModel(),
                    graph_factory=FakeGraph,
                )


if __name__ == "__main__":
    unittest.main()
