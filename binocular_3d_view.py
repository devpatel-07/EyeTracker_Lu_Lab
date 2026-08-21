from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import time

import numpy as np


Coordinate3D = tuple[float, float, float]
Segment3D = tuple[Coordinate3D, Coordinate3D]


BACKGROUND_COLOR = (246, 246, 243)
GRID_COLOR = (222, 222, 218)
LEFT_COLOR = (205, 118, 35)
RIGHT_COLOR = (55, 92, 215)
PUPIL_COLOR = (24, 24, 24)
LENS_COLOR = (105, 105, 100)


@dataclass(frozen=True)
class SideSceneState:
    eye_center_mm: Coordinate3D
    pupil_center_mm: Coordinate3D
    camera_lens_mm: Coordinate3D
    gaze_segment_mm: Segment3D
    camera_axis_segment_mm: Segment3D


@dataclass(frozen=True)
class _MeshHandle:
    mesh: object
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray


def _finite_number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value, name):
    result = _finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _bgr_to_rgb(color):
    return tuple(component / 255.0 for component in reversed(color))


def _finite_triplet(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        return None
    return result


def build_side_scene_state(
    coordinates,
    camera_yaw_degrees,
    gaze_ray_length_mm,
    camera_axis_length_mm=14.0,
):
    if coordinates is None:
        return None
    values = [
        _finite_triplet(coordinates.shared_eye_center_mm),
        _finite_triplet(coordinates.shared_pupil_center_mm),
        _finite_triplet(coordinates.shared_camera_lens_mm),
        _finite_triplet(coordinates.shared_gaze_direction),
    ]
    if any(value is None for value in values):
        return None
    eye, pupil, lens, gaze = values
    yaw = float(camera_yaw_degrees)
    gaze_length = float(gaze_ray_length_mm)
    axis_length = float(camera_axis_length_mm)
    if not all(math.isfinite(value) for value in (yaw, gaze_length, axis_length)):
        return None
    if gaze_length <= 0 or axis_length <= 0:
        return None
    gaze_end = tuple(
        origin + direction * gaze_length
        for origin, direction in zip(eye, gaze)
    )
    yaw_radians = math.radians(yaw)
    camera_end = (
        lens[0] + math.sin(yaw_radians) * axis_length,
        lens[1],
        lens[2] + math.cos(yaw_radians) * axis_length,
    )
    return SideSceneState(
        eye_center_mm=eye,
        pupil_center_mm=pupil,
        camera_lens_mm=lens,
        gaze_segment_mm=(eye, gaze_end),
        camera_axis_segment_mm=(lens, camera_end),
    )


class Binocular3DView:
    def __init__(
        self,
        eye_radius_mm=12.0,
        gaze_ray_length_mm=80.0,
        max_fps=30.0,
        left_camera_yaw_degrees=35.0,
        right_camera_yaw_degrees=-35.0,
        *,
        open3d_module=None,
        clock=time.monotonic,
    ):
        self.eye_radius_mm = _positive_finite(eye_radius_mm, "eye radius")
        self.gaze_ray_length_mm = _positive_finite(
            gaze_ray_length_mm, "gaze ray length"
        )
        self.max_fps = _positive_finite(max_fps, "viewer maximum FPS")
        self.left_yaw = _finite_number(left_camera_yaw_degrees, "left camera yaw")
        self.right_yaw = _finite_number(
            right_camera_yaw_degrees, "right camera yaw"
        )
        self._clock = clock
        self._render_interval_s = 1.0 / self.max_fps
        self._last_render_s = -math.inf
        self._active = True
        self._destroyed = False
        if open3d_module is None:
            try:
                open3d_module = importlib.import_module("open3d")
            except ImportError as error:
                raise RuntimeError(
                    "3D viewer requires requirements-visualization.txt"
                ) from error
        self._o3d = open3d_module
        self._line_colors = {}
        self._mesh_handles = {}
        self._initial_zoom_pending = True
        self._visualizer = self._create_window()
        try:
            self._dynamic = self._create_scene()
        except Exception:
            self.close()
            raise

    def _create_window(self):
        visualizer = self._o3d.visualization.Visualizer()
        if not visualizer.create_window(
            window_name="Binocular 3D Geometry", width=900, height=700
        ):
            raise RuntimeError("could not initialize Open3D viewer")
        render_option = visualizer.get_render_option()
        render_option.background_color = np.asarray(
            _bgr_to_rgb(BACKGROUND_COLOR), dtype=float
        )
        render_option.line_width = 2.0
        render_option.point_size = 5.0
        return visualizer

    def _create_scene(self):
        coordinate_frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=20.0
        )
        self._visualizer.add_geometry(coordinate_frame)
        self._visualizer.add_geometry(self._create_grid())

        dynamic = {}
        for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
            eye, eye_handle = self._create_mesh(self.eye_radius_mm, color)
            pupil, pupil_handle = self._create_mesh(2.0, PUPIL_COLOR)
            camera, camera_handle = self._create_mesh(3.0, LENS_COLOR)
            gaze = self._create_line(color)
            camera_axis = self._create_line(color)
            dynamic[side] = {
                "eye": eye,
                "pupil": pupil,
                "camera": camera,
                "gaze": gaze,
                "camera_axis": camera_axis,
            }
            self._mesh_handles[side] = {
                "eye": eye_handle,
                "pupil": pupil_handle,
                "camera": camera_handle,
            }
        return dynamic

    def _create_grid(self):
        points = []
        lines = []
        for x in range(-100, 101, 10):
            lines.append((len(points), len(points) + 1))
            points.extend(((x, 0.0, -100.0), (x, 0.0, 50.0)))
        for z in range(-100, 51, 10):
            lines.append((len(points), len(points) + 1))
            points.extend(((-100.0, 0.0, z), (100.0, 0.0, z)))
        grid = self._o3d.geometry.LineSet()
        grid.points = self._o3d.utility.Vector3dVector(points)
        grid.lines = self._o3d.utility.Vector2iVector(lines)
        grid.colors = self._o3d.utility.Vector3dVector(
            [_bgr_to_rgb(GRID_COLOR)] * len(lines)
        )
        return grid

    def _create_mesh(self, radius, color):
        mesh = self._o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(_bgr_to_rgb(color))
        handle = _MeshHandle(
            mesh=mesh,
            vertices=np.asarray(mesh.vertices, dtype=float).copy(),
            triangles=np.asarray(mesh.triangles, dtype=int).copy(),
            normals=np.asarray(mesh.vertex_normals, dtype=float).copy(),
        )
        self._visualizer.add_geometry(mesh)
        self._hide_mesh(handle)
        self._visualizer.update_geometry(mesh)
        return mesh, handle

    def _create_line(self, color):
        line = self._o3d.geometry.LineSet()
        self._line_colors[id(line)] = _bgr_to_rgb(color)
        self._show_line(line, ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        self._visualizer.add_geometry(line)
        self._hide_line(line)
        self._visualizer.update_geometry(line)
        return line

    def _show_mesh(self, handle, center):
        center = np.asarray(center, dtype=float)
        handle.mesh.vertices = self._o3d.utility.Vector3dVector(
            handle.vertices + center
        )
        handle.mesh.triangles = self._o3d.utility.Vector3iVector(handle.triangles)
        handle.mesh.vertex_normals = self._o3d.utility.Vector3dVector(handle.normals)

    def _hide_mesh(self, handle):
        handle.mesh.vertices = self._o3d.utility.Vector3dVector(np.empty((0, 3)))
        handle.mesh.triangles = self._o3d.utility.Vector3iVector(np.empty((0, 3)))
        handle.mesh.vertex_normals = self._o3d.utility.Vector3dVector(
            np.empty((0, 3))
        )

    def _show_line(self, line_set, segment):
        line_set.points = self._o3d.utility.Vector3dVector(segment)
        line_set.lines = self._o3d.utility.Vector2iVector(((0, 1),))
        line_set.colors = self._o3d.utility.Vector3dVector(
            (self._line_colors[id(line_set)],)
        )

    def _hide_line(self, line_set):
        line_set.points = self._o3d.utility.Vector3dVector(np.empty((0, 3)))
        line_set.lines = self._o3d.utility.Vector2iVector(np.empty((0, 2)))
        line_set.colors = self._o3d.utility.Vector3dVector(np.empty((0, 3)))

    def _apply_side_state(self, side, state):
        dynamic = self._dynamic[side]
        handles = self._mesh_handles[side]
        if state is None:
            self._hide_mesh(handles["eye"])
            self._hide_mesh(handles["pupil"])
            self._hide_mesh(handles["camera"])
            self._hide_line(dynamic["gaze"])
            self._hide_line(dynamic["camera_axis"])
        else:
            self._show_mesh(handles["eye"], state.eye_center_mm)
            self._show_mesh(handles["pupil"], state.pupil_center_mm)
            self._show_mesh(handles["camera"], state.camera_lens_mm)
            self._show_line(dynamic["gaze"], state.gaze_segment_mm)
            self._show_line(dynamic["camera_axis"], state.camera_axis_segment_mm)
        for geometry in dynamic.values():
            self._visualizer.update_geometry(geometry)

    def update(self, left_coordinates, right_coordinates):
        if not self._active:
            return False
        try:
            if not self._visualizer.poll_events():
                self.close()
                return False
            now = float(self._clock())
            if now - self._last_render_s < self._render_interval_s:
                return True
            states = {
                "left": build_side_scene_state(
                    left_coordinates, self.left_yaw, self.gaze_ray_length_mm
                ),
                "right": build_side_scene_state(
                    right_coordinates, self.right_yaw, self.gaze_ray_length_mm
                ),
            }
            for side, state in states.items():
                self._apply_side_state(side, state)
            if self._initial_zoom_pending:
                self._visualizer.get_view_control().set_zoom(0.9)
                self._initial_zoom_pending = False
            self._visualizer.update_renderer()
            self._last_render_s = now
            return True
        except Exception as error:
            print("3D viewer disabled:", error)
            self.close()
            return False

    def close(self):
        if self._destroyed:
            return
        self._active = False
        self._destroyed = True
        try:
            self._visualizer.destroy_window()
        except Exception:
            pass
