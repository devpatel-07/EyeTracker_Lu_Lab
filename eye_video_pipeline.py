from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from binocular_geometry import BinocularCoordinates, to_shared_coordinates
from camera_geometry import rotated_camera_matrix, undistort_ellipse
from pupil_dilation_graph import PupilDilationGraph
from saccade_velocity import (
    SaccadeEvent,
    SaccadeVelocityGraph,
    SaccadeVelocityTracker,
)
from temporal_eye_model import EyeModelResult, TemporalEyeModel
from video_frame_transform import FrameTransform


@dataclass(frozen=True)
class EyeCameraCalibration:
    path: Path
    rotation: str
    raw_camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    raw_image_size: tuple[int, int]
    source_frame_size: tuple[int, int]
    frame_transform: FrameTransform
    camera_matrix: np.ndarray
    expected_frame_size: tuple[int, int]

    @classmethod
    def load(cls, path, rotation, frame_transform=None):
        path = Path(path)
        try:
            with np.load(path) as calibration:
                raw_camera_matrix = np.asarray(
                    calibration["camera_matrix"],
                    dtype=np.float64,
                )
                distortion_coefficients = np.asarray(
                    calibration["dist_coeffs"],
                    dtype=np.float64,
                )
                raw_image_size = tuple(
                    int(value) for value in calibration["image_size"]
                )
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(
                f"Could not load camera calibration {path}: {error}"
            ) from error

        if raw_camera_matrix.shape != (3, 3) or not np.all(
            np.isfinite(raw_camera_matrix)
        ):
            raise ValueError("camera matrix must be a finite 3x3 matrix")
        if distortion_coefficients.size == 0 or not np.all(
            np.isfinite(distortion_coefficients)
        ):
            raise ValueError("distortion coefficients must be finite")
        if len(raw_image_size) != 2 or any(
            value <= 0 for value in raw_image_size
        ):
            raise ValueError("calibration image size must be positive")
        if frame_transform is None:
            frame_transform = FrameTransform()
        elif not isinstance(frame_transform, FrameTransform):
            raise ValueError("frame transform must be a FrameTransform")

        source_camera_matrix = rotated_camera_matrix(
            raw_camera_matrix,
            raw_image_size,
            rotation,
        )
        source_frame_size = (
            raw_image_size if rotation == "none" else raw_image_size[::-1]
        )
        camera_matrix = frame_transform.transform_camera_matrix(
            source_camera_matrix,
            source_frame_size,
        )
        expected_frame_size = frame_transform.output_size(source_frame_size)

        return cls(
            path=path,
            rotation=rotation,
            raw_camera_matrix=raw_camera_matrix,
            distortion_coefficients=distortion_coefficients,
            raw_image_size=raw_image_size,
            source_frame_size=source_frame_size,
            frame_transform=frame_transform,
            camera_matrix=camera_matrix,
            expected_frame_size=expected_frame_size,
        )


@dataclass(frozen=True)
class EyeFrameOutput:
    display: np.ndarray
    timestamp_s: float
    model_result: EyeModelResult | None
    coordinates: BinocularCoordinates | None
    pupil_diameter_mm: float | None
    blink: bool
    saccade_velocity_deg_s: float | None = None
    saccade_event: SaccadeEvent | None = None


def validate_synchronized_captures(
    left_capture,
    right_capture,
    fps_tolerance=0.01,
):
    for side, capture in (
        ("left", left_capture),
        ("right", right_capture),
    ):
        if capture is None or not capture.isOpened():
            raise ValueError(f"Could not open {side} video")

    left_fps = float(left_capture.get(cv2.CAP_PROP_FPS))
    right_fps = float(right_capture.get(cv2.CAP_PROP_FPS))
    if not all(
        math.isfinite(value) and value > 0
        for value in (left_fps, right_fps)
    ):
        raise ValueError("video frame rates must be finite and positive")
    if not math.isclose(
        left_fps,
        right_fps,
        rel_tol=0.0,
        abs_tol=float(fps_tolerance),
    ):
        raise ValueError(
            "video frame rates do not match: "
            f"left={left_fps}, right={right_fps}"
        )
    return left_fps


def read_frame_pair(left_capture, right_capture):
    left_ok, left_frame = left_capture.read()
    right_ok, right_frame = right_capture.read()
    if not left_ok or not right_ok:
        return None
    return left_frame, right_frame


class EyeVideoPipeline:
    def __init__(
        self,
        side,
        capture,
        calibration,
        pupil_detector,
        eye_center_separation_mm,
        *,
        rect=None,
        camera_yaw_degrees=0.0,
        min_confidence=0.60,
        graph_history_seconds=10.0,
        graph_baseline_seconds=5.0,
        saccade_velocity_threshold_deg_s=30.0,
        saccade_min_amplitude_deg=1.0,
        temporal_model_factory=TemporalEyeModel,
        graph_factory=PupilDilationGraph,
        saccade_factory=SaccadeVelocityTracker,
    ):
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.capture = capture
        self.calibration = calibration
        self.pupil_detector = pupil_detector
        try:
            self.eye_center_separation_mm = float(eye_center_separation_mm)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "eye-center separation must be finite and positive"
            ) from error
        if (
            not math.isfinite(self.eye_center_separation_mm)
            or self.eye_center_separation_mm <= 0
        ):
            raise ValueError(
                "eye-center separation must be finite and positive"
            )
        try:
            self.camera_yaw_degrees = float(camera_yaw_degrees)
        except (TypeError, ValueError) as error:
            raise ValueError("camera yaw must be finite") from error
        if not math.isfinite(self.camera_yaw_degrees):
            raise ValueError("camera yaw must be finite")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (width, height)
        if self.frame_size != calibration.expected_frame_size:
            raise ValueError(
                f"{side} video size {self.frame_size} does not match "
                f"calibrated {calibration.rotation} size "
                f"{calibration.expected_frame_size}"
            )
        self.rect = (
            self._default_rect(width, height)
            if rect is None
            else self._validated_rect(rect, width, height)
        )

        self.temporal_model = temporal_model_factory(
            camera_matrix=calibration.camera_matrix,
            resolution=self.frame_size,
            min_confidence=min_confidence,
        )
        self.graph = graph_factory(
            history_seconds=graph_history_seconds,
            baseline_seconds=graph_baseline_seconds,
        )
        self.saccades = saccade_factory(
            velocity_threshold_deg_s=saccade_velocity_threshold_deg_s,
            min_amplitude_deg=saccade_min_amplitude_deg,
            history_seconds=graph_history_seconds,
        )
        self.saccade_graph = SaccadeVelocityGraph(self.saccades)
        self._fallback_start_s = time.monotonic()
        self._last_timestamp_s = None
        self._graph_started = False

    @staticmethod
    def _default_rect(width, height):
        x = 100 if width > 200 else 0
        y = 100 if height > 200 else 0
        return {
            "x": x,
            "y": y,
            "w": width - x,
            "h": min(600, height - y),
        }

    @staticmethod
    def _validated_rect(rect, frame_width, frame_height):
        if not isinstance(rect, dict) or set(rect) != {"x", "y", "w", "h"}:
            raise ValueError("rect must contain exactly x, y, w, and h")
        try:
            result = {name: int(rect[name]) for name in ("x", "y", "w", "h")}
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("rect values must be finite integers") from error

        x, y, width, height = (
            result["x"],
            result["y"],
            result["w"],
            result["h"],
        )
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("rect coordinates and dimensions are invalid")
        if x + width > frame_width or y + height > frame_height:
            raise ValueError("rect must fit completely inside the video frame")
        return result

    def _timestamp(self):
        capture_timestamp_ms = float(
            self.capture.get(cv2.CAP_PROP_POS_MSEC)
        )
        if math.isfinite(capture_timestamp_ms) and capture_timestamp_ms >= 0:
            capture_timestamp_s = capture_timestamp_ms / 1000.0
            if (
                self._last_timestamp_s is None
                or capture_timestamp_s > self._last_timestamp_s
            ):
                self._last_timestamp_s = capture_timestamp_s
                return capture_timestamp_s

        fallback = time.monotonic() - self._fallback_start_s
        if self._last_timestamp_s is not None and fallback <= self._last_timestamp_s:
            fallback = math.nextafter(self._last_timestamp_s, math.inf)
        self._last_timestamp_s = fallback
        return fallback

    @staticmethod
    def _diameter(result):
        if result is None or not result.ready:
            return None
        if result.corrected_diameter_mm is not None:
            return result.corrected_diameter_mm
        return result.raw_diameter_mm

    def process_frame(self, frame):
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            raise ValueError(
                f"{self.side} frame size does not match video configuration"
            )

        timestamp_s = self._timestamp()
        display, ellipse, blink, confidence = (
            self.pupil_detector.detect_with_confidence(
                frame,
                self.rect,
                debug=False,
            )
        )

        model_result = None
        coordinates = None
        if not blink:
            geometry_ellipse = undistort_ellipse(
                ellipse,
                self.calibration.raw_camera_matrix,
                self.calibration.distortion_coefficients,
                self.calibration.raw_image_size,
                self.calibration.rotation,
                frame_transform=self.calibration.frame_transform,
            )
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            model_result = self.temporal_model.update(
                ellipse=geometry_ellipse,
                confidence=confidence,
                timestamp=timestamp_s,
                grayscale=grayscale,
            )
            if model_result.ready:
                source_eye_center = (
                    self.calibration.frame_transform.to_source_camera_axes(
                        model_result.eye_center_mm
                    )
                )
                source_pupil_center = (
                    self.calibration.frame_transform.to_source_camera_axes(
                        model_result.pupil_center_mm
                    )
                )
                coordinates = to_shared_coordinates(
                    self.side,
                    self.eye_center_separation_mm,
                    source_eye_center,
                    source_pupil_center,
                    camera_yaw_degrees=self.camera_yaw_degrees,
                )

        pupil_diameter_mm = self._diameter(model_result)
        if model_result is not None and model_result.ready and not self._graph_started:
            self.graph.start(timestamp_s)
            self._graph_started = True
        if self._graph_started:
            self.graph.add_sample(timestamp_s, pupil_diameter_mm)

        # A blink or an unready model feeds None so the tracker drops any
        # movement in progress instead of reading the recovery as a saccade.
        saccade_event = self.saccades.add_sample(
            timestamp_s,
            coordinates.shared_gaze_direction if coordinates is not None else None,
        )

        self._draw_status(display, coordinates is not None)
        return EyeFrameOutput(
            display=display,
            timestamp_s=timestamp_s,
            model_result=model_result,
            coordinates=coordinates,
            pupil_diameter_mm=pupil_diameter_mm,
            blink=bool(blink),
            saccade_velocity_deg_s=self.saccades.latest_velocity_deg_s,
            saccade_event=saccade_event,
        )

    def _draw_status(self, display, ready):
        label = f"{self.side.capitalize()} eye"
        status = "3D model active" if ready else "Building 3D model"
        color = (0, 200, 0) if ready else (0, 180, 255)
        cv2.putText(
            display,
            label,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            status,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def render_graph(self):
        return self.graph.render()

    def render_saccade_graph(self):
        return self.saccade_graph.render()
