from __future__ import annotations

from dataclasses import dataclass
import math


Coordinate3D = tuple[float, float, float]


@dataclass(frozen=True)
class BinocularCoordinates:
    local_eye_center_mm: Coordinate3D
    local_pupil_center_mm: Coordinate3D
    shared_eye_center_mm: Coordinate3D
    shared_pupil_center_mm: Coordinate3D
    shared_camera_lens_mm: Coordinate3D
    shared_gaze_direction: Coordinate3D


def _finite_triplet(value, name):
    try:
        coordinate = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite 3D coordinate") from error

    if len(coordinate) != 3 or not all(
        math.isfinite(component) for component in coordinate
    ):
        raise ValueError(f"{name} must be a finite 3D coordinate")
    return coordinate


def rotate_about_y(coordinate, yaw_degrees):
    coordinate = _finite_triplet(coordinate, "coordinate")
    try:
        yaw_degrees = float(yaw_degrees)
    except (TypeError, ValueError) as error:
        raise ValueError("camera yaw must be finite") from error
    if not math.isfinite(yaw_degrees):
        raise ValueError("camera yaw must be finite")

    yaw = math.radians(yaw_degrees)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x, y, z = coordinate
    return (
        cosine * x + sine * z,
        y,
        -sine * x + cosine * z,
    )


def _camera_to_shared_axes(coordinate):
    x, y, z = coordinate
    return (x, -y, z)


def to_shared_coordinates(
    side,
    separation_mm,
    local_eye_center_mm,
    local_pupil_center_mm,
    camera_yaw_degrees=0.0,
):
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    separation_mm = float(separation_mm)
    if not math.isfinite(separation_mm) or separation_mm <= 0:
        raise ValueError("separation_mm must be finite and positive")

    local_eye = _finite_triplet(local_eye_center_mm, "local eye center")
    local_pupil = _finite_triplet(
        local_pupil_center_mm,
        "local pupil center",
    )

    anchor_x = separation_mm / 2.0
    if side == "left":
        anchor_x = -anchor_x
    shared_eye = (anchor_x, 0.0, 0.0)
    local_relative_pupil = _camera_to_shared_axes(
        tuple(
            pupil_component - eye_component
            for pupil_component, eye_component in zip(local_pupil, local_eye)
        )
    )
    relative_pupil = rotate_about_y(
        local_relative_pupil,
        camera_yaw_degrees,
    )
    gaze_length = math.sqrt(
        sum(component * component for component in relative_pupil)
    )
    if not math.isfinite(gaze_length) or gaze_length <= 0:
        raise ValueError("gaze vector must have nonzero finite length")
    gaze_direction = tuple(
        component / gaze_length for component in relative_pupil
    )
    shared_pupil = tuple(
        eye_component + pupil_component
        for eye_component, pupil_component in zip(shared_eye, relative_pupil)
    )
    rotated_local_eye = rotate_about_y(
        _camera_to_shared_axes(local_eye),
        camera_yaw_degrees,
    )
    shared_camera_lens = tuple(
        eye_component - local_component
        for eye_component, local_component in zip(
            shared_eye,
            rotated_local_eye,
        )
    )

    return BinocularCoordinates(
        local_eye_center_mm=local_eye,
        local_pupil_center_mm=local_pupil,
        shared_eye_center_mm=shared_eye,
        shared_pupil_center_mm=shared_pupil,
        shared_camera_lens_mm=shared_camera_lens,
        shared_gaze_direction=gaze_direction,
    )
