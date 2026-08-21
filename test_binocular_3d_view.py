import math
from types import SimpleNamespace
import unittest

import numpy as np

from binocular_geometry import BinocularCoordinates, to_shared_coordinates
from binocular_3d_view import Binocular3DView, build_side_scene_state


class FakeRenderOption:
    def __init__(self):
        self.background_color = None
        self.line_width = None
        self.point_size = None


class FakeMesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=float)
        self.triangles = np.empty((0, 3), dtype=int)
        self.vertex_normals = np.empty((0, 3), dtype=float)
        self.color = None

    def paint_uniform_color(self, color):
        self.color = np.asarray(color, dtype=float)

    def compute_vertex_normals(self):
        if len(self.vertices):
            self.vertex_normals = np.tile(
                np.array([[0.0, 1.0, 0.0]]), (len(self.vertices), 1)
            )


class FakeLineSet:
    def __init__(self):
        self.points = np.empty((0, 3), dtype=float)
        self.lines = np.empty((0, 2), dtype=int)
        self.colors = np.empty((0, 3), dtype=float)


class FakeTriangleMeshFactory:
    @staticmethod
    def create_sphere(radius):
        mesh = FakeMesh()
        mesh.vertices = np.array([[0.0, 0.0, 0.0]])
        mesh.triangles = np.array([[0, 0, 0]])
        mesh.vertex_normals = np.array([[0.0, 1.0, 0.0]])
        mesh.radius = radius
        return mesh

    @staticmethod
    def create_coordinate_frame(size):
        mesh = FakeMesh()
        mesh.vertices = np.array([[0.0, 0.0, 0.0]])
        mesh.size = size
        return mesh


class FakeVisualizer:
    def __init__(self):
        self.create_window_calls = []
        self.geometries = []
        self.added_geometry_point_counts = []
        self.render_option = FakeRenderOption()
        self.update_count = 0
        self.poll_count = 0
        self.render_count = 0
        self.destroy_count = 0
        self.reset_count = 0
        self.zoom_values = []
        self.view_events = []
        self.poll_result = True

    def create_window(self, **kwargs):
        self.create_window_calls.append(kwargs)
        return True

    def add_geometry(self, geometry):
        self.geometries.append(geometry)
        coordinates = (
            geometry.points if hasattr(geometry, "points") else geometry.vertices
        )
        self.added_geometry_point_counts.append(len(coordinates))

    def get_render_option(self):
        return self.render_option

    def get_view_control(self):
        return self

    def update_geometry(self, geometry):
        self.update_count += 1

    def poll_events(self):
        self.poll_count += 1
        self.view_events.append("poll")
        return self.poll_result

    def update_renderer(self):
        self.render_count += 1
        self.view_events.append("render")

    def destroy_window(self):
        self.destroy_count += 1

    def reset_view_point(self, _reset):
        self.reset_count += 1

    def set_zoom(self, value):
        self.zoom_values.append(value)
        self.view_events.append("zoom")


def make_fake_open3d():
    visualizer = FakeVisualizer()
    return SimpleNamespace(
        visualization=SimpleNamespace(Visualizer=lambda: visualizer),
        geometry=SimpleNamespace(
            TriangleMesh=FakeTriangleMeshFactory,
            LineSet=FakeLineSet,
        ),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=float),
            Vector2iVector=lambda values: np.asarray(values, dtype=int),
            Vector3iVector=lambda values: np.asarray(values, dtype=int),
        ),
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class SideSceneStateTests(unittest.TestCase):
    def setUp(self):
        self.left = to_shared_coordinates(
            "left",
            64.0,
            (-10.0, 4.0, 35.0),
            (-8.0, 3.0, 24.0),
            camera_yaw_degrees=35.0,
        )

    def test_builds_full_xyz_markers_and_rays(self):
        state = build_side_scene_state(self.left, 35.0, 80.0)

        self.assertEqual(state.eye_center_mm, self.left.shared_eye_center_mm)
        self.assertEqual(state.pupil_center_mm, self.left.shared_pupil_center_mm)
        self.assertEqual(state.camera_lens_mm, self.left.shared_camera_lens_mm)
        self.assertEqual(state.gaze_segment_mm[0], state.eye_center_mm)
        expected_gaze_end = tuple(
            eye + direction * 80.0
            for eye, direction in zip(
                state.eye_center_mm,
                self.left.shared_gaze_direction,
            )
        )
        np.testing.assert_allclose(state.gaze_segment_mm[1], expected_gaze_end)
        expected_camera_end = (
            state.camera_lens_mm[0] + math.sin(math.radians(35.0)) * 14.0,
            state.camera_lens_mm[1],
            state.camera_lens_mm[2] + math.cos(math.radians(35.0)) * 14.0,
        )
        np.testing.assert_allclose(
            state.camera_axis_segment_mm[1],
            expected_camera_end,
        )

    def test_none_or_nonfinite_coordinates_hide_side(self):
        self.assertIsNone(build_side_scene_state(None, 35.0, 80.0))
        invalid = BinocularCoordinates(
            local_eye_center_mm=(0.0, 0.0, 1.0),
            local_pupil_center_mm=(0.0, 0.0, 1.0),
            shared_eye_center_mm=(math.nan, 0.0, 0.0),
            shared_pupil_center_mm=(0.0, 0.0, -1.0),
            shared_camera_lens_mm=(0.0, 0.0, 1.0),
            shared_gaze_direction=(0.0, 0.0, -1.0),
        )
        self.assertIsNone(build_side_scene_state(invalid, 35.0, 80.0))


class Binocular3DViewTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.open3d = make_fake_open3d()
        self.left = to_shared_coordinates(
            "left", 64.0, (-10.0, 4.0, 35.0), (-8.0, 3.0, 24.0), 35.0
        )
        self.right = to_shared_coordinates(
            "right", 64.0, (10.0, 4.0, 35.0), (8.0, 3.0, 24.0), -35.0
        )

    def make_viewer(self, **overrides):
        settings = {
            "eye_radius_mm": 12.0,
            "gaze_ray_length_mm": 80.0,
            "max_fps": 30.0,
            "left_camera_yaw_degrees": 35.0,
            "right_camera_yaw_degrees": -35.0,
            "open3d_module": self.open3d,
            "clock": self.clock,
        }
        settings.update(overrides)
        return Binocular3DView(**settings)

    def test_updates_both_sides_without_resetting_view(self):
        viewer = self.make_viewer()

        self.assertTrue(viewer.update(self.left, self.right))

        self.assertGreater(viewer._visualizer.update_count, 0)
        self.assertEqual(viewer._visualizer.reset_count, 0)

    def test_registers_every_geometry_with_nonempty_bounds(self):
        viewer = self.make_viewer()

        self.assertTrue(viewer._visualizer.added_geometry_point_counts)
        self.assertTrue(
            all(
                count > 0
                for count in viewer._visualizer.added_geometry_point_counts
            )
        )

    def test_initial_zoom_waits_for_first_render_and_is_not_reapplied(self):
        viewer = self.make_viewer()
        self.assertEqual(viewer._visualizer.zoom_values, [])
        self.assertEqual(viewer._visualizer.view_events, [])

        viewer.update(self.left, self.right)

        self.assertEqual(viewer._visualizer.zoom_values, [0.9])
        self.assertEqual(
            viewer._visualizer.view_events,
            ["poll", "zoom", "render"],
        )

        self.clock.value += 1.0
        viewer.update(self.left, self.right)

        self.assertEqual(viewer._visualizer.zoom_values, [0.9])
        self.assertEqual(
            viewer._visualizer.view_events,
            ["poll", "zoom", "render", "poll", "render"],
        )
        self.assertEqual(viewer._visualizer.reset_count, 0)

    def test_throttling_still_polls_events(self):
        viewer = self.make_viewer()
        viewer.update(self.left, self.right)
        rendered = viewer._visualizer.render_count
        polled = viewer._visualizer.poll_count

        self.clock.value += 0.001
        viewer.update(self.left, self.right)

        self.assertEqual(viewer._visualizer.render_count, rendered)
        self.assertEqual(viewer._visualizer.poll_count, polled + 1)

    def test_missing_side_clears_stale_dynamic_geometry(self):
        viewer = self.make_viewer()
        viewer.update(self.left, self.right)

        self.clock.value += 1.0
        viewer.update(self.left, None)

        self.assertEqual(len(viewer._dynamic["right"]["pupil"].vertices), 0)
        self.assertEqual(len(viewer._dynamic["right"]["gaze"].points), 0)

    def test_closed_window_returns_false_and_close_is_idempotent(self):
        viewer = self.make_viewer()
        viewer._visualizer.poll_result = False

        self.assertFalse(viewer.update(self.left, self.right))
        viewer.close()
        viewer.close()

        self.assertEqual(viewer._visualizer.destroy_count, 1)

    def test_constructor_closes_window_when_scene_creation_fails(self):
        failing_open3d = make_fake_open3d()
        mesh_factory = failing_open3d.geometry.TriangleMesh
        failing_open3d.geometry.TriangleMesh = SimpleNamespace(
            create_sphere=mesh_factory.create_sphere,
            create_coordinate_frame=lambda size: (_ for _ in ()).throw(
                RuntimeError("scene creation failed")
            ),
        )
        visualizer = failing_open3d.visualization.Visualizer()

        with self.assertRaisesRegex(RuntimeError, "scene creation failed"):
            self.make_viewer(open3d_module=failing_open3d)

        self.assertEqual(visualizer.destroy_count, 1)

    def test_constructor_rejects_invalid_settings(self):
        invalid_positive_settings = {
            "eye_radius_mm": ("eye radius", (0, -1, math.nan, "invalid")),
            "gaze_ray_length_mm": (
                "gaze ray length",
                (0, -1, math.inf, "invalid"),
            ),
            "max_fps": ("viewer maximum FPS", (0, -1, -math.inf, "invalid")),
        }
        for setting, (name, values) in invalid_positive_settings.items():
            for value in values:
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.make_viewer(**{setting: value})

        for setting, name in (
            ("left_camera_yaw_degrees", "left camera yaw"),
            ("right_camera_yaw_degrees", "right camera yaw"),
        ):
            for value in (math.nan, math.inf, "invalid"):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.make_viewer(**{setting: value})


if __name__ == "__main__":
    unittest.main()
