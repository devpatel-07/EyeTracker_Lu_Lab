from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np


PYE3D_REFERENCE_EYE_RADIUS_MM = 10.392304845413264


@dataclass(frozen=True)
class EyeModelResult:
    ready: bool
    eye_center_mm: tuple[float, float, float] | None
    pupil_center_mm: tuple[float, float, float] | None
    raw_diameter_mm: float | None
    corrected_diameter_mm: float | None
    confidence: float
    update_time_ms: float
    raw_result: dict | None


def _create_pye3d_detector(focal_length, resolution, min_confidence):
    try:
        from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
    except ImportError as error:
        raise RuntimeError(
            "pye3d is required; install requirements-eye-model.txt"
        ) from error

    try:
        camera = CameraModel(
            focal_length=focal_length,
            resolution=resolution,
        )
        return Detector3D(
            camera=camera,
            threshold_swirski=max(0.0, min_confidence - 1e-6),
            threshold_short_term=min_confidence,
            threshold_long_term=min_confidence,
            long_term_mode=DetectorMode.blocking,
        )
    except Exception as error:
        raise RuntimeError(f"Could not initialize pye3d: {error}") from error


def _finite_triplet(value):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        return None
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        return None
    return result


class TemporalEyeModel:
    def __init__(
        self,
        camera_matrix,
        resolution,
        min_confidence=0.60,
        eye_radius_mm=12.0,
        detector_factory: Callable | None = None,
    ):
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")

        self.width, self.height = (int(value) for value in resolution)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resolution values must be positive")

        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

        self.min_confidence = float(min_confidence)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        self.eye_radius_mm = float(eye_radius_mm)
        if not math.isfinite(self.eye_radius_mm) or self.eye_radius_mm <= 0:
            raise ValueError("eye_radius_mm must be finite and positive")
        self._scale = self.eye_radius_mm / PYE3D_REFERENCE_EYE_RADIUS_MM

        focal_length = (self.fx + self.fy) / 2.0
        factory = detector_factory or _create_pye3d_detector
        try:
            self._detector = factory(
                focal_length,
                (self.width, self.height),
                self.min_confidence,
            )
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Could not initialize temporal eye model: {error}"
            ) from error

        self._last_timestamp = None

    @staticmethod
    def _canonicalize_ellipse(ellipse):
        try:
            (center_x, center_y), (axis_0, axis_1), angle = ellipse
            values = tuple(
                float(value)
                for value in (center_x, center_y, axis_0, axis_1, angle)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("ellipse must use the OpenCV ellipse format") from error

        if not all(math.isfinite(value) for value in values):
            raise ValueError("ellipse values must be finite")
        center_x, center_y, axis_0, axis_1, angle = values
        if axis_0 <= 0 or axis_1 <= 0:
            raise ValueError("ellipse axes must be positive")

        if axis_0 <= axis_1:
            axes = (axis_0, axis_1)
        else:
            axes = (axis_1, axis_0)
            angle += 90.0
        return (center_x, center_y), axes, angle % 180.0

    def _empty_result(self, confidence=0.0, update_time_ms=0.0, raw_result=None):
        return EyeModelResult(
            ready=False,
            eye_center_mm=None,
            pupil_center_mm=None,
            raw_diameter_mm=None,
            corrected_diameter_mm=None,
            confidence=float(confidence),
            update_time_ms=float(update_time_ms),
            raw_result=raw_result,
        )

    def update(self, ellipse, confidence, timestamp, grayscale):
        grayscale = np.asarray(grayscale)
        if grayscale.ndim != 2 or grayscale.shape != (self.height, self.width):
            raise ValueError(
                f"grayscale frame size must be {(self.width, self.height)}"
            )
        if grayscale.dtype != np.uint8:
            raise ValueError("grayscale frame must have dtype uint8")

        confidence = float(confidence)
        timestamp = float(timestamp)
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        center, axes, angle = self._canonicalize_ellipse(ellipse)
        if confidence < self.min_confidence:
            return self._empty_result(confidence=confidence)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("accepted observation timestamp must increase")

        model_center = (
            center[0] + self.width / 2.0 - self.cx,
            center[1] + self.height / 2.0 - self.cy,
        )
        datum = {
            "ellipse": {
                "center": model_center,
                "axes": axes,
                "angle": angle,
            },
            "diameter": axes[1],
            "location": model_center,
            "confidence": confidence,
            "timestamp": timestamp,
            "norm_pos": (
                model_center[0] / self.width,
                1.0 - model_center[1] / self.height,
            ),
            "method": "custom-ml",
        }

        started = time.perf_counter()
        try:
            raw_result = self._detector.update_and_detect(
                datum,
                grayscale,
                apply_refraction_correction=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"pye3d update failed at timestamp {timestamp}: {error}"
            ) from error
        update_time_ms = (time.perf_counter() - started) * 1000.0
        self._last_timestamp = timestamp

        if not isinstance(raw_result, dict):
            return self._empty_result(
                confidence=confidence,
                update_time_ms=update_time_ms,
            )

        sphere = raw_result.get("sphere", {})
        circle = raw_result.get("circle_3d", {})
        eye_center = _finite_triplet(sphere.get("center"))
        pupil_center = _finite_triplet(circle.get("center"))
        try:
            diameter = float(raw_result.get("diameter_3d"))
            radius_diameter = 2.0 * float(circle.get("radius"))
            result_confidence = float(raw_result.get("confidence", confidence))
        except (TypeError, ValueError):
            return self._empty_result(
                confidence=confidence,
                update_time_ms=update_time_ms,
                raw_result=raw_result,
            )

        diameter_agrees = math.isclose(
            diameter,
            radius_diameter,
            rel_tol=0.02,
            abs_tol=0.05,
        )
        scaled_eye_center = (
            tuple(value * self._scale for value in eye_center)
            if eye_center is not None
            else None
        )
        scaled_pupil_center = (
            tuple(value * self._scale for value in pupil_center)
            if pupil_center is not None
            else None
        )
        scaled_diameter = diameter * self._scale
        minimum_eye_depth = 15.0 * self._scale
        maximum_eye_depth = 75.0 * self._scale

        angles_valid = True
        if raw_result.get("phi") is not None and raw_result.get("theta") is not None:
            phi_degrees = math.degrees(float(raw_result["phi"]))
            theta_degrees = math.degrees(float(raw_result["theta"]))
            angles_valid = (
                -90.0 <= phi_degrees + 90.0 <= 90.0
                and -80.0 <= theta_degrees - 90.0 <= 80.0
            )
        ready = (
            scaled_eye_center is not None
            and scaled_pupil_center is not None
            and minimum_eye_depth
            <= scaled_eye_center[2]
            <= maximum_eye_depth
            and scaled_pupil_center[2] > 0
            and math.isfinite(diameter)
            and 1.0 <= scaled_diameter <= 9.0
            and diameter_agrees
            and math.isfinite(result_confidence)
            and result_confidence >= self.min_confidence
            and angles_valid
        )
        if not ready:
            return self._empty_result(
                confidence=(
                    result_confidence
                    if math.isfinite(result_confidence)
                    else confidence
                ),
                update_time_ms=update_time_ms,
                raw_result=raw_result,
            )

        return EyeModelResult(
            ready=True,
            eye_center_mm=scaled_eye_center,
            pupil_center_mm=scaled_pupil_center,
            raw_diameter_mm=scaled_diameter,
            corrected_diameter_mm=None,
            confidence=result_confidence,
            update_time_ms=update_time_ms,
            raw_result=raw_result,
        )
