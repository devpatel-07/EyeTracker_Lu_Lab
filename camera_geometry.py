import math

import cv2
import numpy as np

from video_frame_transform import FrameTransform


VALID_ROTATIONS = {"none", "clockwise", "counterclockwise"}


def _validate_rotation(rotation):
    if rotation not in VALID_ROTATIONS:
        raise ValueError(f"Unsupported video rotation: {rotation}")


def _to_raw_orientation(points, raw_image_size, rotation):
    width, height = raw_image_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if rotation == "none":
        return points.copy()
    if rotation == "clockwise":
        return np.column_stack((points[:, 1], height - 1 - points[:, 0]))
    return np.column_stack((width - 1 - points[:, 1], points[:, 0]))


def _from_raw_orientation(points, raw_image_size, rotation):
    width, height = raw_image_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if rotation == "none":
        return points.copy()
    if rotation == "clockwise":
        return np.column_stack((height - 1 - points[:, 1], points[:, 0]))
    return np.column_stack((points[:, 1], width - 1 - points[:, 0]))


def rotated_camera_matrix(camera_matrix, raw_image_size, rotation="none"):
    _validate_rotation(rotation)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if rotation == "none":
        return camera_matrix.copy()

    width, height = raw_image_size
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    if rotation == "clockwise":
        rotated_cx = height - 1 - cy
        rotated_cy = cx
    else:
        rotated_cx = cy
        rotated_cy = width - 1 - cx

    return np.array(
        [[fy, 0.0, rotated_cx], [0.0, fx, rotated_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def undistort_points(
    points,
    camera_matrix,
    distortion_coefficients,
    raw_image_size,
    rotation="none",
    frame_transform=None,
):
    _validate_rotation(rotation)
    if frame_transform is None:
        frame_transform = FrameTransform()
    source_size = raw_image_size if rotation == "none" else raw_image_size[::-1]
    source_points = frame_transform.inverse_points(points, source_size)
    raw_points = _to_raw_orientation(source_points, raw_image_size, rotation)
    corrected_raw = cv2.undistortPoints(
        raw_points.reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
        P=np.asarray(camera_matrix, dtype=np.float64),
    ).reshape(-1, 2)
    corrected_source = _from_raw_orientation(
        corrected_raw,
        raw_image_size,
        rotation,
    )
    return frame_transform.forward_points(corrected_source, source_size)


def distort_points(
    points,
    camera_matrix,
    distortion_coefficients,
    raw_image_size,
    rotation="none",
    frame_transform=None,
):
    _validate_rotation(rotation)
    if frame_transform is None:
        frame_transform = FrameTransform()
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    source_size = raw_image_size if rotation == "none" else raw_image_size[::-1]
    corrected_source = frame_transform.inverse_points(points, source_size)
    corrected_raw = _to_raw_orientation(
        corrected_source,
        raw_image_size,
        rotation,
    )

    homogeneous = np.column_stack(
        (corrected_raw, np.ones(len(corrected_raw), dtype=np.float64))
    )
    normalized = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    object_points = np.column_stack(
        (
            normalized[:, 0] / normalized[:, 2],
            normalized[:, 1] / normalized[:, 2],
            np.ones(len(normalized), dtype=np.float64),
        )
    )
    distorted_raw, _ = cv2.projectPoints(
        object_points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        np.asarray(distortion_coefficients, dtype=np.float64),
    )
    distorted_source = _from_raw_orientation(
        distorted_raw.reshape(-1, 2), raw_image_size, rotation
    )
    return frame_transform.forward_points(distorted_source, source_size)


def undistort_ellipse(
    ellipse,
    camera_matrix,
    distortion_coefficients,
    raw_image_size,
    rotation="none",
    samples=72,
    frame_transform=None,
):
    (center_x, center_y), (axis_width, axis_height), angle_degrees = ellipse
    if axis_width <= 0 or axis_height <= 0 or samples < 5:
        raise ValueError("Ellipse axes must be positive and samples must be at least 5")

    semi_width = axis_width / 2.0
    semi_height = axis_height / 2.0
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    parameters = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)

    local_x = semi_width * np.cos(parameters)
    local_y = semi_height * np.sin(parameters)
    points = np.column_stack(
        (
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        )
    )
    corrected_points = undistort_points(
        points,
        camera_matrix,
        distortion_coefficients,
        raw_image_size,
        rotation,
        frame_transform,
    )
    return cv2.fitEllipse(corrected_points.astype(np.float32).reshape(-1, 1, 2))
