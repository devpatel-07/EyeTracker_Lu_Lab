import inspect
import subprocess
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class Fake3DView:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = []
        self.close_count = 0
        self.update_result = True
        self.__class__.instances.append(self)

    def update(self, left_coordinates, right_coordinates):
        self.updates.append((left_coordinates, right_coordinates))
        return self.update_result

    def close(self):
        self.close_count += 1


class PupilDetectionImportTests(unittest.TestCase):
    def setUp(self):
        Fake3DView.instances = []

    def test_visualization_requirements_pin_open3d(self):
        import pupil_detection

        requirements = (
            Path(pupil_detection.__file__).with_name("requirements-visualization.txt")
        ).read_text(encoding="ascii").splitlines()
        self.assertEqual(requirements, ["open3d==0.19.0"])

    def test_import_does_not_launch_file_dialog(self):
        project_dir = Path(__file__).resolve().parent
        code = """
import tkinter

class RejectTk:
    def __init__(self):
        raise RuntimeError('file dialog launched during import')

tkinter.Tk = RejectTk
import pupil_detection
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_does_not_load_optional_open3d_dependency(self):
        project_dir = Path(__file__).resolve().parent
        code = """
import sys
import pupil_detection

if 'open3d' in sys.modules:
    raise RuntimeError('pupil_detection imported open3d eagerly')
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pupil_graph_configuration_is_exposed(self):
        import pupil_detection

        self.assertEqual(pupil_detection.PUPIL_GRAPH_HISTORY_SECONDS, 10.0)
        self.assertEqual(pupil_detection.PUPIL_GRAPH_BASELINE_SECONDS, 5.0)

    def test_temporal_eye_model_configuration_is_exposed(self):
        import pupil_detection

        self.assertEqual(pupil_detection.MIN_PYE3D_CONFIDENCE, 0.60)

    def test_binocular_configuration_is_exposed(self):
        import pupil_detection

        self.assertGreater(pupil_detection.EYE_CENTER_SEPARATION_MM, 0.0)
        self.assertEqual(
            pupil_detection.LEFT_CALIBRATION_PATH,
            pupil_detection.RIGHT_CALIBRATION_PATH,
        )
        self.assertEqual(pupil_detection.LEFT_VIDEO_ROTATION, "clockwise")
        self.assertEqual(pupil_detection.RIGHT_VIDEO_ROTATION, "clockwise")
        self.assertEqual(pupil_detection.LEFT_FRAME_ROTATION, "none")
        self.assertFalse(pupil_detection.LEFT_FRAME_FLIP_HORIZONTAL)
        self.assertFalse(pupil_detection.LEFT_FRAME_FLIP_VERTICAL)
        self.assertEqual(pupil_detection.RIGHT_FRAME_ROTATION, "none")
        self.assertFalse(pupil_detection.RIGHT_FRAME_FLIP_HORIZONTAL)
        self.assertFalse(pupil_detection.RIGHT_FRAME_FLIP_VERTICAL)
        self.assertEqual(pupil_detection.LEFT_CAMERA_YAW_DEGREES, 35.0)
        self.assertEqual(pupil_detection.RIGHT_CAMERA_YAW_DEGREES, -35.0)
        self.assertEqual(
            pupil_detection.LEFT_EYE_RECT,
            {"x": 0, "y": 100, "w": 980, "h": 600},
        )
        self.assertEqual(
            pupil_detection.RIGHT_EYE_RECT,
            {"x": 100, "y": 100, "w": 980, "h": 600},
        )

    def test_module_contains_only_active_runtime_geometry(self):
        import pupil_detection

        source = inspect.getsource(pupil_detection)

        self.assertEqual(source.count("EyeVideoPipeline("), 2)
        retired_names = (
            "def detect_pupil(",
            "def compute_average_intersection(",
            "def compute_3D_eye_center(",
            "def compute_3D_pupil_center(",
            "def create_radius_calibrator(",
            "USE_ML_PUPIL_DETECTOR",
        )
        for retired_name in retired_names:
            with self.subTest(retired_name=retired_name):
                self.assertNotIn(retired_name, source)

    def test_main_builds_two_pipelines_with_one_detector(self):
        import pupil_detection

        left_capture = MagicMock()
        right_capture = MagicMock()
        detector = object()
        pipelines = []

        class FakePipeline:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.frame_size = (1080, 1920)
                pipelines.append(self)

            def process_frame(self, _frame):
                raise AssertionError("no frames should be processed")

            def render_graph(self):
                raise AssertionError("no graph should be rendered")

        with (
            patch.object(pupil_detection, "ENABLE_3D_VIEWER", False),
            patch.object(
                pupil_detection,
                "access_file",
                side_effect=["left.mp4", "right.mp4"],
            ) as access_file,
            patch.object(
                pupil_detection.cv2,
                "VideoCapture",
                side_effect=[left_capture, right_capture],
            ),
            patch.object(
                pupil_detection,
                "validate_synchronized_captures",
            ),
            patch.object(
                pupil_detection.MLPupilDetector,
                "from_checkpoint",
                return_value=detector,
            ),
            patch.object(
                pupil_detection,
                "EyeCameraCalibration",
            ) as calibration_type,
            patch.object(
                pupil_detection,
                "EyeVideoPipeline",
                FakePipeline,
            ),
            patch.object(
                pupil_detection,
                "read_frame_pair",
                return_value=None,
            ),
            patch.object(pupil_detection.cv2, "destroyAllWindows"),
        ):
            calibration_type.load.side_effect = ["left-cal", "right-cal"]
            pupil_detection.main()

        self.assertEqual(
            [call.args[0] for call in access_file.call_args_list],
            ["left", "right"],
        )
        self.assertEqual([item.kwargs["side"] for item in pipelines], ["left", "right"])
        self.assertIs(pipelines[0].kwargs["pupil_detector"], detector)
        self.assertIs(pipelines[1].kwargs["pupil_detector"], detector)
        self.assertEqual(pipelines[0].kwargs["camera_yaw_degrees"], 35.0)
        self.assertEqual(pipelines[1].kwargs["camera_yaw_degrees"], -35.0)
        self.assertEqual(
            pipelines[0].kwargs["rect"],
            pupil_detection.LEFT_EYE_RECT,
        )
        self.assertEqual(
            pipelines[1].kwargs["rect"],
            pupil_detection.RIGHT_EYE_RECT,
        )
        left_capture.release.assert_called_once_with()
        right_capture.release.assert_called_once_with()

    def test_main_applies_independent_runtime_transforms_to_each_eye(self):
        import pupil_detection
        from video_frame_transform import FrameTransform

        left_capture = MagicMock()
        right_capture = MagicMock()
        detector = object()
        transformed_captures = []
        pipelines = []

        class RecordingTransformedCapture:
            def __init__(self, capture, frame_transform):
                self.capture = capture
                self.frame_transform = frame_transform
                transformed_captures.append(self)

            def release(self):
                self.capture.release()

        class FakePipeline:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.frame_size = (1080, 1920)
                pipelines.append(self)

        with (
            patch.object(pupil_detection, "ENABLE_3D_VIEWER", False),
            patch.object(
                pupil_detection,
                "LEFT_FRAME_ROTATION",
                "clockwise",
            ),
            patch.object(
                pupil_detection,
                "LEFT_FRAME_FLIP_HORIZONTAL",
                True,
            ),
            patch.object(
                pupil_detection,
                "RIGHT_FRAME_ROTATION",
                "180",
            ),
            patch.object(
                pupil_detection,
                "RIGHT_FRAME_FLIP_VERTICAL",
                True,
            ),
            patch.object(
                pupil_detection,
                "access_file",
                side_effect=["left.mp4", "right.mp4"],
            ),
            patch.object(
                pupil_detection.cv2,
                "VideoCapture",
                side_effect=[left_capture, right_capture],
            ),
            patch.object(
                pupil_detection,
                "TransformedVideoCapture",
                RecordingTransformedCapture,
            ),
            patch.object(
                pupil_detection,
                "validate_synchronized_captures",
            ),
            patch.object(
                pupil_detection.MLPupilDetector,
                "from_checkpoint",
                return_value=detector,
            ),
            patch.object(
                pupil_detection,
                "EyeCameraCalibration",
            ) as calibration_type,
            patch.object(pupil_detection, "EyeVideoPipeline", FakePipeline),
            patch.object(pupil_detection, "read_frame_pair", return_value=None),
            patch.object(pupil_detection.cv2, "destroyAllWindows"),
        ):
            calibration_type.load.side_effect = ["left-cal", "right-cal"]
            pupil_detection.main()

        expected_left_transform = FrameTransform(
            rotation="clockwise",
            flip_horizontal=True,
        )
        expected_right_transform = FrameTransform(
            rotation="180",
            flip_vertical=True,
        )
        self.assertEqual(
            [capture.capture for capture in transformed_captures],
            [left_capture, right_capture],
        )
        self.assertEqual(
            [capture.frame_transform for capture in transformed_captures],
            [expected_left_transform, expected_right_transform],
        )
        self.assertEqual(
            calibration_type.load.call_args_list,
            [
                (
                    (pupil_detection.LEFT_CALIBRATION_PATH,
                     pupil_detection.LEFT_VIDEO_ROTATION,
                     expected_left_transform),
                    {},
                ),
                (
                    (pupil_detection.RIGHT_CALIBRATION_PATH,
                     pupil_detection.RIGHT_VIDEO_ROTATION,
                     expected_right_transform),
                    {},
                ),
            ],
        )
        self.assertEqual(
            [pipeline.kwargs["capture"] for pipeline in pipelines],
            transformed_captures,
        )
        left_capture.release.assert_called_once_with()
        right_capture.release.assert_called_once_with()

    def test_main_releases_both_sources_when_runtime_transform_is_invalid(self):
        import pupil_detection

        left_capture = MagicMock()
        right_capture = MagicMock()

        with (
            patch.object(
                pupil_detection,
                "LEFT_FRAME_ROTATION",
                "not-a-rotation",
            ),
            patch.object(
                pupil_detection,
                "access_file",
                side_effect=["left.mp4", "right.mp4"],
            ),
            patch.object(
                pupil_detection.cv2,
                "VideoCapture",
                side_effect=[left_capture, right_capture],
            ),
            patch.object(pupil_detection, "TransformedVideoCapture") as wrapper_type,
            patch.object(pupil_detection.cv2, "destroyAllWindows"),
            patch("builtins.print") as print_mock,
        ):
            pupil_detection.main()

        wrapper_type.assert_not_called()
        left_capture.release.assert_called_once_with()
        right_capture.release.assert_called_once_with()
        self.assertEqual(print_mock.call_count, 1)
        self.assertEqual(
            print_mock.call_args.args[0],
            "Could not run binocular eye tracker:",
        )
        error = print_mock.call_args.args[1]
        self.assertIsInstance(error, ValueError)
        self.assertIn(
            "rotation must be none, clockwise, counterclockwise, or 180",
            str(error),
        )

    def run_main_with_paired_frames(self, frame_pairs):
        import pupil_detection
        from binocular_geometry import to_shared_coordinates

        left_capture = MagicMock()
        right_capture = MagicMock()
        detector = object()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        display = np.zeros((8, 8, 3), dtype=np.uint8)
        graph = np.zeros((6, 6, 3), dtype=np.uint8)
        pipelines = []

        left_coordinates = to_shared_coordinates(
            "left",
            63.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
            camera_yaw_degrees=35.0,
        )
        right_coordinates = to_shared_coordinates(
            "right",
            63.0,
            (10.0, 4.0, 35.0),
            (8.0, 3.0, 24.0),
            camera_yaw_degrees=-35.0,
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                self.side = kwargs["side"]
                self.frame_size = (1080, 1920)
                self.processed_frames = []
                pipelines.append(self)

            def process_frame(self, input_frame):
                self.processed_frames.append(input_frame)
                coordinates = (
                    left_coordinates
                    if self.side == "left"
                    else right_coordinates
                )
                return SimpleNamespace(
                    display=display,
                    timestamp_s=0.0,
                    coordinates=coordinates,
                    pupil_diameter_mm=3.2,
                    model_result=SimpleNamespace(
                        confidence=0.9,
                        update_time_ms=1.5,
                    ),
                )

            def render_graph(self):
                return graph

        with (
            patch.object(
                pupil_detection,
                "access_file",
                side_effect=["left.mp4", "right.mp4"],
            ),
            patch.object(
                pupil_detection.cv2,
                "VideoCapture",
                side_effect=[left_capture, right_capture],
            ),
            patch.object(
                pupil_detection,
                "validate_synchronized_captures",
            ),
            patch.object(
                pupil_detection.MLPupilDetector,
                "from_checkpoint",
                return_value=detector,
            ),
            patch.object(
                pupil_detection,
                "EyeCameraCalibration",
            ) as calibration_type,
            patch.object(
                pupil_detection,
                "EyeVideoPipeline",
                FakePipeline,
            ),
            patch.object(
                pupil_detection,
                "read_frame_pair",
                side_effect=[*frame_pairs, None],
            ),
            patch.object(pupil_detection.cv2, "imshow") as imshow,
            patch.object(pupil_detection.cv2, "waitKey", return_value=-1),
            patch.object(pupil_detection.cv2, "destroyAllWindows"),
        ):
            calibration_type.load.side_effect = ["left-cal", "right-cal"]
            pupil_detection.main()

        return SimpleNamespace(
            left_coordinates=left_coordinates,
            right_coordinates=right_coordinates,
            pipelines=pipelines,
            imshow=imshow,
        )

    def run_main_with_one_paired_frame(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return self.run_main_with_paired_frames([(frame, frame)])

    def test_main_does_not_construct_viewer_when_disabled(self):
        import pupil_detection

        with patch.object(pupil_detection, "ENABLE_3D_VIEWER", False), patch.object(
            pupil_detection, "Binocular3DView"
        ) as viewer_type:
            self.run_main_with_one_paired_frame()

        viewer_type.assert_not_called()

    def test_main_updates_and_closes_enabled_viewer(self):
        import pupil_detection

        with patch.object(pupil_detection, "ENABLE_3D_VIEWER", True), patch.object(
            pupil_detection, "Binocular3DView", Fake3DView
        ):
            result = self.run_main_with_one_paired_frame()

        viewer = Fake3DView.instances[-1]
        self.assertEqual(viewer.kwargs["eye_radius_mm"], 12.0)
        self.assertEqual(viewer.kwargs["max_fps"], 30.0)
        self.assertEqual(viewer.kwargs["left_camera_yaw_degrees"], 35.0)
        self.assertEqual(viewer.kwargs["right_camera_yaw_degrees"], -35.0)
        self.assertEqual(
            viewer.updates,
            [(result.left_coordinates, result.right_coordinates)],
        )
        self.assertEqual(viewer.close_count, 1)
        self.assertFalse(
            any(
                call.args[0] == "binocular_top_view"
                for call in result.imshow.call_args_list
            )
        )

    def test_main_keeps_processing_after_viewer_deactivates(self):
        import pupil_detection

        class StopsAfterFirstUpdate(Fake3DView):
            def update(self, left_coordinates, right_coordinates):
                self.update_result = False
                return super().update(left_coordinates, right_coordinates)

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch.object(pupil_detection, "ENABLE_3D_VIEWER", True), patch.object(
            pupil_detection, "Binocular3DView", StopsAfterFirstUpdate
        ):
            result = self.run_main_with_paired_frames([(frame, frame), (frame, frame)])

        viewer = StopsAfterFirstUpdate.instances[-1]
        self.assertEqual([len(pipeline.processed_frames) for pipeline in result.pipelines], [2, 2])
        self.assertEqual(
            viewer.updates,
            [(result.left_coordinates, result.right_coordinates)],
        )
        self.assertEqual(viewer.close_count, 0)

    def test_main_logs_lens_and_gaze_geometry(self):
        import pupil_detection

        source = inspect.getsource(pupil_detection.main)
        self.assertIn("shared_camera_lens_mm", source)
        self.assertIn("shared_gaze_direction", source)

if __name__ == "__main__":
    unittest.main()
