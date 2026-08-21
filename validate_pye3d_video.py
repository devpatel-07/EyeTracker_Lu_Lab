import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from camera_geometry import rotated_camera_matrix, undistort_ellipse
from ml_pupil_inference import MLPupilDetector
from temporal_eye_model import TemporalEyeModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_DIR / "eye_test_custom_new_clockwise.mp4"
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "pupil_unet_best.pt"
DEFAULT_CALIBRATION = PROJECT_DIR / "camera_calibration.npz"
DEFAULT_MIN_CONFIDENCE = 0.60


def _finite_coordinates(result):
    coordinates = result.eye_center_mm + result.pupil_center_mm
    return all(math.isfinite(value) for value in coordinates)


def _distribution(values, include_p95=False):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }
    if include_p95:
        summary["p95"] = float(np.percentile(array, 95))
    return summary


def _pye3d_out_of_range_reasons(raw_results):
    reasons = {
        "eye_center_x": 0,
        "eye_center_y": 0,
        "eye_center_z": 0,
        "pupil_diameter": 0,
        "gaze_angles": 0,
    }
    for raw_result in raw_results:
        sphere = raw_result.get("sphere", {})
        center = sphere.get("center")
        if center is not None and len(center) == 3:
            x, y, z = (float(value) for value in center)
            reasons["eye_center_x"] += not -15.0 <= x <= 15.0
            reasons["eye_center_y"] += not -10.0 <= y <= 10.0
            reasons["eye_center_z"] += not 15.0 <= z <= 75.0

        diameter = raw_result.get("diameter_3d")
        if diameter is not None:
            reasons["pupil_diameter"] += not 1.0 <= float(diameter) <= 9.0

        phi = raw_result.get("phi")
        theta = raw_result.get("theta")
        if phi is not None and theta is not None:
            phi_degrees = math.degrees(float(phi))
            theta_degrees = math.degrees(float(theta))
            angles_valid = (
                -90.0 <= phi_degrees + 90.0 <= 90.0
                and -80.0 <= theta_degrees - 90.0 <= 80.0
            )
            reasons["gaze_angles"] += not angles_valid
    return reasons


def summarize_results(results):
    ready_results = [result for result in results if result.ready]
    finite_failures = sum(
        not _finite_coordinates(result) for result in ready_results
    )
    diameters = [
        result.corrected_diameter_mm
        if result.corrected_diameter_mm is not None
        else result.raw_diameter_mm
        for result in ready_results
    ]
    timings = [result.update_time_ms for result in results]

    raw_results = [
        result.raw_result
        for result in results
        if isinstance(result.raw_result, dict)
    ]
    out_of_range = sum(
        float(raw_result.get("model_confidence", 0.0)) < 0.5
        for raw_result in raw_results
    )
    raw_diameters = [
        float(raw_result["diameter_3d"])
        for raw_result in raw_results
        if raw_result.get("diameter_3d") is not None
        and math.isfinite(float(raw_result["diameter_3d"]))
    ]
    raw_depths = [
        float(raw_result["sphere"]["center"][2])
        for raw_result in raw_results
        if isinstance(raw_result.get("sphere"), dict)
        and raw_result["sphere"].get("center") is not None
        and len(raw_result["sphere"]["center"]) == 3
        and math.isfinite(float(raw_result["sphere"]["center"][2]))
    ]

    return {
        "accepted_detections": len(results),
        "ready_outputs": len(ready_results),
        "finite_coordinate_failures": finite_failures,
        "pye3d_updates": len(raw_results),
        "pre_pye3d_rejections": len(results) - len(raw_results),
        "model_out_of_range_outputs": out_of_range,
        "pye3d_out_of_range_reasons": _pye3d_out_of_range_reasons(
            raw_results
        ),
        "observation_confidence": _distribution(
            [result.confidence for result in results]
        ),
        "diameter_mm": _distribution(diameters),
        "eye_center_z_mm": _distribution(
            [result.eye_center_mm[2] for result in ready_results]
        ),
        "pupil_center_z_mm": _distribution(
            [result.pupil_center_mm[2] for result in ready_results]
        ),
        "raw_model_diameter_mm": _distribution(raw_diameters),
        "raw_eye_center_z_mm": _distribution(raw_depths),
        "update_time_ms": _distribution(timings, include_p95=True),
    }


def run_validation(
    video_path,
    checkpoint_path,
    calibration_path,
    rotation="clockwise",
    max_frames=300,
):
    with np.load(calibration_path) as calibration:
        raw_camera_matrix = calibration["camera_matrix"].astype(np.float64)
        distortion = calibration["dist_coeffs"].astype(np.float64)
        raw_image_size = tuple(
            int(value) for value in calibration["image_size"]
        )
    camera_matrix = rotated_camera_matrix(
        raw_camera_matrix,
        raw_image_size,
        rotation,
    )

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_width, raw_height = raw_image_size
    expected_size = (
        (raw_width, raw_height)
        if rotation == "none"
        else (raw_height, raw_width)
    )
    if (width, height) != expected_size:
        video.release()
        raise ValueError(
            f"Video size {(width, height)} does not match {expected_size}"
        )

    detector = MLPupilDetector.from_checkpoint(
        checkpoint_path,
        device="auto",
        threshold=0.5,
    )
    eye_model = TemporalEyeModel(
        camera_matrix=camera_matrix,
        resolution=(width, height),
        min_confidence=DEFAULT_MIN_CONFIDENCE,
    )
    rect = {"x": 100, "y": 100, "w": width - 100, "h": 600}
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    results = []
    frames_processed = 0
    blink_frames = 0
    while max_frames <= 0 or frames_processed < max_frames:
        ongoing, frame = video.read()
        if not ongoing:
            break

        timestamp = frames_processed / fps
        _, ellipse, blink, confidence = detector.detect_with_confidence(
            frame,
            rect,
            debug=False,
        )
        frames_processed += 1
        if blink:
            blink_frames += 1
            continue

        corrected_ellipse = undistort_ellipse(
            ellipse,
            raw_camera_matrix,
            distortion,
            raw_image_size,
            rotation,
        )
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results.append(
            eye_model.update(
                corrected_ellipse,
                confidence,
                timestamp,
                grayscale,
            )
        )

    video.release()
    summary = summarize_results(results)
    summary.update(
        {
            "frames_processed": frames_processed,
            "blink_frames": blink_frames,
            "model_became_ready": summary["ready_outputs"] > 0,
        }
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate pye3d geometry and timing on a recorded eye video"
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--rotation",
        choices=("none", "clockwise", "counterclockwise"),
        default="clockwise",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_validation(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        rotation=args.rotation,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
