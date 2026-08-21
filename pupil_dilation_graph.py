import math

import cv2
import numpy as np


class PupilDilationGraph:
    def __init__(
        self,
        history_seconds=10.0,
        baseline_seconds=5.0,
        width=900,
        height=520,
    ):
        self.history_seconds = float(history_seconds)
        self.baseline_seconds = float(baseline_seconds)
        self.width = int(width)
        self.height = int(height)

        self._started = False
        self._start_timestamp_s = None
        self._baseline_start_s = None
        self._baseline_values = []
        self._baseline_mm = None
        self._history = []
        self._last_timestamp_s = None
        self._latest_percentage_change = None

    @property
    def baseline_mm(self):
        return self._baseline_mm

    @property
    def baseline_progress(self):
        if self._baseline_mm is not None:
            return 1.0
        if self._baseline_start_s is None or self._last_timestamp_s is None:
            return 0.0
        elapsed = self._last_timestamp_s - self._baseline_start_s
        return min(1.0, max(0.0, elapsed / self.baseline_seconds))

    @property
    def latest_percentage_change(self):
        return self._latest_percentage_change

    @property
    def history(self):
        return tuple(self._history)

    def start(self, timestamp_s):
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            return False
        if not self._started:
            self._started = True
            self._start_timestamp_s = timestamp_s
        return True

    def add_sample(self, timestamp_s, diameter_mm):
        if not self._started:
            return False

        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            return False
        if (
            self._last_timestamp_s is not None
            and timestamp_s <= self._last_timestamp_s
        ):
            return False

        value = self._validated_diameter(diameter_mm)
        self._last_timestamp_s = timestamp_s
        self._history.append((timestamp_s, value))
        self._trim_history(timestamp_s)

        if value is not None and self._baseline_start_s is None:
            self._baseline_start_s = timestamp_s

        if value is not None and self._baseline_mm is None:
            if timestamp_s - self._baseline_start_s <= self.baseline_seconds:
                self._baseline_values.append(value)

        if (
            self._baseline_mm is None
            and self._baseline_start_s is not None
            and timestamp_s - self._baseline_start_s >= self.baseline_seconds
            and self._baseline_values
        ):
            self._baseline_mm = float(np.median(self._baseline_values))

        if value is not None and self._baseline_mm is not None:
            self._latest_percentage_change = (
                100.0 * (value - self._baseline_mm) / self._baseline_mm
            )
        else:
            self._latest_percentage_change = None

        return True

    @staticmethod
    def _validated_diameter(diameter_mm):
        if diameter_mm is None:
            return None
        try:
            diameter_mm = float(diameter_mm)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(diameter_mm) or diameter_mm <= 0:
            return None
        return diameter_mm

    def _trim_history(self, latest_timestamp_s):
        cutoff = latest_timestamp_s - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.pop(0)

    def render(self):
        canvas = np.full(
            (self.height, self.width, 3),
            (247, 247, 245),
            dtype=np.uint8,
        )
        top_rect, bottom_rect = self._plot_rectangles()

        self._draw_header(canvas)

        diameter_values = list(self._history)
        diameter_range = self._diameter_range()
        self._draw_plot(
            canvas,
            top_rect,
            diameter_values,
            diameter_range,
            title="Pupil diameter (mm)",
            color=(64, 145, 52),
            reference_value=self._baseline_mm,
            reference_color=(55, 145, 225),
        )

        percentage_values = [
            (
                timestamp_s,
                None
                if value is None or self._baseline_mm is None
                else 100.0 * (value - self._baseline_mm) / self._baseline_mm,
            )
            for timestamp_s, value in self._history
        ]
        self._draw_plot(
            canvas,
            bottom_rect,
            percentage_values,
            self._percentage_range(percentage_values),
            title="Change from baseline (%)",
            color=(185, 105, 35),
            reference_value=0.0,
            reference_color=(150, 150, 150),
        )

        return canvas

    def _plot_rectangles(self):
        left = 72
        right = self.width - 24
        top = 86
        bottom_margin = 24
        gap = 48
        available_height = self.height - top - bottom_margin - gap
        plot_height = max(40, available_height // 2)
        top_rect = (left, top, right, top + plot_height)
        bottom_top = top + plot_height + gap
        bottom_rect = (left, bottom_top, right, bottom_top + plot_height)
        return top_rect, bottom_rect

    def _draw_header(self, canvas):
        cv2.putText(
            canvas,
            "Pupil dilation",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )

        if not self._started:
            status = "Waiting for eye-radius calibration"
        elif self._baseline_mm is None:
            status = f"Collecting baseline: {self.baseline_progress * 100:.0f}%"
        else:
            latest_value = self._latest_valid_value()
            if latest_value is None or self._latest_percentage_change is None:
                status = f"Baseline: {self._baseline_mm:.2f} mm | Latest: unavailable"
            else:
                status = (
                    f"Baseline: {self._baseline_mm:.2f} mm | "
                    f"Latest: {latest_value:.2f} mm "
                    f"({self._latest_percentage_change:+.1f}%)"
                )

        cv2.putText(
            canvas,
            status,
            (18, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (75, 75, 75),
            1,
            cv2.LINE_AA,
        )

    def _draw_plot(
        self,
        canvas,
        rect,
        values,
        value_range,
        title,
        color,
        reference_value=None,
        reference_color=(160, 160, 160),
    ):
        left, top, right, bottom = rect
        cv2.rectangle(canvas, (left, top), (right, bottom), (255, 255, 255), -1)

        value_min, value_max = value_range
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(round(bottom - fraction * (bottom - top)))
            cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)

        if reference_value is not None and value_min <= reference_value <= value_max:
            reference_y = self._map_value_to_y(
                reference_value,
                value_min,
                value_max,
                top,
                bottom,
            )
            cv2.line(
                canvas,
                (left, reference_y),
                (right, reference_y),
                reference_color,
                1,
                cv2.LINE_AA,
            )

        time_min, time_max = self._time_range()
        previous_point = None
        previous_timestamp = None
        for timestamp_s, value in values:
            if value is None:
                previous_point = None
                previous_timestamp = None
                continue

            x = int(
                round(
                    left
                    + (timestamp_s - time_min)
                    * (right - left)
                    / (time_max - time_min)
                )
            )
            y = self._map_value_to_y(value, value_min, value_max, top, bottom)
            point = (x, y)

            if (
                previous_point is not None
                and timestamp_s - previous_timestamp <= 0.5
            ):
                cv2.line(
                    canvas,
                    previous_point,
                    point,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            previous_point = point
            previous_timestamp = timestamp_s

        cv2.rectangle(canvas, (left, top), (right, bottom), (175, 175, 175), 1)
        cv2.putText(
            canvas,
            title,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

        self._draw_y_label(canvas, value_max, left - 8, top + 4)
        self._draw_y_label(
            canvas,
            (value_min + value_max) / 2.0,
            left - 8,
            (top + bottom) // 2 + 4,
        )
        self._draw_y_label(canvas, value_min, left - 8, bottom + 4)

    @staticmethod
    def _draw_y_label(canvas, value, right_x, baseline_y):
        label = f"{value:.1f}"
        (text_width, _), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            1,
        )
        cv2.putText(
            canvas,
            label,
            (right_x - text_width, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    def _time_range(self):
        if self._last_timestamp_s is None:
            start = (
                self._start_timestamp_s
                if self._start_timestamp_s is not None
                else 0.0
            )
            return start, start + self.history_seconds

        start = (
            self._start_timestamp_s
            if self._start_timestamp_s is not None
            else self._last_timestamp_s
        )
        elapsed = self._last_timestamp_s - start
        if elapsed < self.history_seconds:
            return start, start + self.history_seconds
        return (
            self._last_timestamp_s - self.history_seconds,
            self._last_timestamp_s,
        )

    def _diameter_range(self):
        values = [value for _, value in self._history if value is not None]
        if self._baseline_mm is not None:
            values.append(self._baseline_mm)
        if not values:
            return 1.5, 8.0

        value_min = min(values)
        value_max = max(values)
        span = value_max - value_min
        padding = max(0.25, span * 0.15)
        return max(0.0, value_min - padding), value_max + padding

    @staticmethod
    def _percentage_range(values):
        finite_values = [value for _, value in values if value is not None]
        if not finite_values:
            return -5.0, 5.0
        limit = max(5.0, max(abs(value) for value in finite_values) * 1.15)
        return -limit, limit

    @staticmethod
    def _map_value_to_y(value, value_min, value_max, top, bottom):
        fraction = (value - value_min) / (value_max - value_min)
        fraction = min(1.0, max(0.0, fraction))
        return int(round(bottom - fraction * (bottom - top)))

    def _latest_valid_value(self):
        if not self._history or self._history[-1][1] is None:
            return None
        return self._history[-1][1]
