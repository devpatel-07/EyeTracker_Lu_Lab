import unittest

import numpy as np

from pupil_dilation_graph import PupilDilationGraph


class PupilDilationGraphStateTests(unittest.TestCase):
    def test_baseline_is_median_of_first_five_seconds(self):
        graph = PupilDilationGraph(baseline_seconds=5.0)
        graph.start(10.0)

        graph.add_sample(10.0, 3.0)
        graph.add_sample(12.0, 5.0)
        graph.add_sample(15.0, 4.0)

        self.assertEqual(graph.baseline_mm, 4.0)
        self.assertEqual(graph.baseline_progress, 1.0)

    def test_percentage_change_uses_completed_baseline(self):
        graph = PupilDilationGraph(baseline_seconds=1.0)
        graph.start(0.0)

        graph.add_sample(0.0, 4.0)
        graph.add_sample(1.0, 4.0)
        graph.add_sample(1.1, 4.4)

        self.assertAlmostEqual(graph.latest_percentage_change, 10.0)

    def test_invalid_measurement_is_stored_as_gap(self):
        graph = PupilDilationGraph()
        graph.start(0.0)

        graph.add_sample(0.1, None)
        graph.add_sample(0.2, float("nan"))
        graph.add_sample(0.3, 0.0)

        self.assertEqual(
            graph.history,
            ((0.1, None), (0.2, None), (0.3, None)),
        )

    def test_history_keeps_only_visible_time_window(self):
        graph = PupilDilationGraph(history_seconds=2.0)
        graph.start(0.0)

        graph.add_sample(0.0, 3.0)
        graph.add_sample(1.0, 3.1)
        graph.add_sample(3.0, 3.2)

        self.assertEqual(graph.history, ((1.0, 3.1), (3.0, 3.2)))

    def test_non_monotonic_timestamp_is_rejected(self):
        graph = PupilDilationGraph()
        graph.start(0.0)
        self.assertTrue(graph.add_sample(1.0, 3.0))

        self.assertFalse(graph.add_sample(1.0, 4.0))
        self.assertFalse(graph.add_sample(0.5, 4.0))
        self.assertEqual(graph.history, ((1.0, 3.0),))


class PupilDilationGraphRenderTests(unittest.TestCase):
    def test_render_returns_nonblank_bgr_canvas(self):
        graph = PupilDilationGraph(width=640, height=400)

        image = graph.render()

        self.assertEqual(image.shape, (400, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(np.unique(image.reshape(-1, 3), axis=0).shape[0], 1)

    def test_visible_time_range_advances_from_zero_start(self):
        graph = PupilDilationGraph(history_seconds=10.0)
        graph.start(0.0)
        graph.add_sample(12.0, 4.0)

        self.assertEqual(graph._time_range(), (2.0, 12.0))

    def test_plot_layout_leaves_header_text_clear(self):
        graph = PupilDilationGraph(width=900, height=520)

        top_rect, _ = graph._plot_rectangles()

        self.assertGreaterEqual(top_rect[1], 80)


if __name__ == "__main__":
    unittest.main()
