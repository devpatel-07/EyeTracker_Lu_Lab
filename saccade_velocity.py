from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SaccadeEvent:
    start_timestamp_s: float
    end_timestamp_s: float
    duration_s: float
    amplitude_deg: float
    peak_velocity_deg_s: float
    mean_velocity_deg_s: float
    sample_count: int


def unit_gaze_vector(value):
    if value is None:
        return None
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(
        math.isfinite(component) for component in vector
    ):
        return None

    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 0:
        return None
    return tuple(component / length for component in vector)


def angular_separation_deg(first, second):
    """Angle between two gaze unit vectors.

    Uses atan2 of the cross-product magnitude against the dot product rather
    than acos of the dot product. Both are equivalent in exact arithmetic, but
    acos loses most of its significant digits for the small angles that appear
    between consecutive frames at high sample rates.
    """
    dot = sum(a * b for a, b in zip(first, second))
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    cross_length = math.sqrt(sum(value * value for value in cross))
    return math.degrees(math.atan2(cross_length, dot))


class SaccadeVelocityTracker:
    """Streaming saccade detector over a sequence of gaze unit vectors.

    Feed one sample per frame with add_sample(). Pass gaze_direction=None for
    blinks or frames where the 3D model was not ready; the tracker discards any
    saccade in progress and suppresses velocity for blink_guard_s afterwards,
    because the frames flanking a blink produce large spurious velocities.
    """

    def __init__(
        self,
        velocity_threshold_deg_s=30.0,
        min_amplitude_deg=1.0,
        min_duration_s=0.0,
        max_sample_gap_s=0.15,
        blink_guard_s=0.10,
        history_seconds=10.0,
    ):
        self.velocity_threshold_deg_s = float(velocity_threshold_deg_s)
        self.min_amplitude_deg = float(min_amplitude_deg)
        self.min_duration_s = float(min_duration_s)
        self.max_sample_gap_s = float(max_sample_gap_s)
        self.blink_guard_s = float(blink_guard_s)
        self.history_seconds = float(history_seconds)

        if self.velocity_threshold_deg_s <= 0:
            raise ValueError("velocity threshold must be positive")
        if self.max_sample_gap_s <= 0:
            raise ValueError("maximum sample gap must be positive")

        self._previous = None
        self._last_timestamp_s = None
        self._blocked_until_s = None

        self._in_saccade = False
        self._saccade_start = None
        self._peak_velocity = 0.0
        self._velocity_sum = 0.0
        self._velocity_count = 0
        self._saccade_end = None

        self._events = []
        self._history = []
        self._latest_velocity = None
        self._latest_event = None
        self._sample_intervals = []

    @property
    def events(self):
        return tuple(self._events)

    @property
    def history(self):
        return tuple(self._history)

    @property
    def latest_velocity_deg_s(self):
        return self._latest_velocity

    @property
    def latest_event(self):
        return self._latest_event

    @property
    def saccade_count(self):
        return len(self._events)

    @property
    def sample_rate_hz(self):
        if not self._sample_intervals:
            return None
        median_interval = float(np.median(self._sample_intervals))
        if median_interval <= 0:
            return None
        return 1.0 / median_interval

    @property
    def saccade_rate_hz(self):
        """Completed saccades per second of tracked time."""
        if not self._events or self._last_timestamp_s is None:
            return None
        elapsed = self._last_timestamp_s - self._events[0].start_timestamp_s
        if elapsed <= 0:
            return None
        return len(self._events) / elapsed

    def add_sample(self, timestamp_s, gaze_direction):
        """Add one frame. Returns the completed SaccadeEvent, or None."""
        try:
            timestamp_s = float(timestamp_s)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(timestamp_s):
            return None
        if (
            self._last_timestamp_s is not None
            and timestamp_s <= self._last_timestamp_s
        ):
            return None

        self._latest_event = None
        gaze = unit_gaze_vector(gaze_direction)

        if gaze is None:
            self._discard_saccade()
            self._previous = None
            self._blocked_until_s = timestamp_s + self.blink_guard_s
            self._record(timestamp_s, None)
            return None

        if (
            self._blocked_until_s is not None
            and timestamp_s < self._blocked_until_s
        ):
            self._previous = (timestamp_s, gaze)
            self._record(timestamp_s, None)
            return None
        self._blocked_until_s = None

        if self._previous is None:
            self._previous = (timestamp_s, gaze)
            self._record(timestamp_s, None)
            return None

        previous_timestamp, previous_gaze = self._previous
        interval_s = timestamp_s - previous_timestamp
        if interval_s <= 0 or interval_s > self.max_sample_gap_s:
            # A dropped frame inflates the interval and would fabricate a
            # saccade, so the gap ends any movement in progress instead.
            self._discard_saccade()
            self._previous = (timestamp_s, gaze)
            self._record(timestamp_s, None)
            return None

        separation_deg = angular_separation_deg(previous_gaze, gaze)
        velocity = separation_deg / interval_s
        self._previous = (timestamp_s, gaze)
        self._sample_intervals.append(interval_s)
        if len(self._sample_intervals) > 300:
            self._sample_intervals.pop(0)
        self._record(timestamp_s, velocity)

        completed = None
        if velocity >= self.velocity_threshold_deg_s:
            if not self._in_saccade:
                self._in_saccade = True
                self._saccade_start = (previous_timestamp, previous_gaze)
                self._peak_velocity = 0.0
                self._velocity_sum = 0.0
                self._velocity_count = 0
            self._peak_velocity = max(self._peak_velocity, velocity)
            self._velocity_sum += velocity
            self._velocity_count += 1
            self._saccade_end = (timestamp_s, gaze)
        elif self._in_saccade:
            completed = self._finalize_saccade()

        self._latest_event = completed
        return completed

    def _record(self, timestamp_s, velocity):
        self._last_timestamp_s = timestamp_s
        self._latest_velocity = velocity
        self._history.append((timestamp_s, velocity))
        cutoff = timestamp_s - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.pop(0)

    def _discard_saccade(self):
        self._in_saccade = False
        self._saccade_start = None
        self._saccade_end = None
        self._peak_velocity = 0.0
        self._velocity_sum = 0.0
        self._velocity_count = 0

    def _finalize_saccade(self):
        start = self._saccade_start
        end = self._saccade_end
        peak_velocity = self._peak_velocity
        mean_velocity = (
            self._velocity_sum / self._velocity_count
            if self._velocity_count
            else 0.0
        )
        sample_count = self._velocity_count
        self._discard_saccade()

        if start is None or end is None:
            return None

        duration_s = end[0] - start[0]
        amplitude_deg = angular_separation_deg(start[1], end[1])
        if duration_s < self.min_duration_s:
            return None
        if amplitude_deg < self.min_amplitude_deg:
            return None

        event = SaccadeEvent(
            start_timestamp_s=start[0],
            end_timestamp_s=end[0],
            duration_s=duration_s,
            amplitude_deg=amplitude_deg,
            peak_velocity_deg_s=peak_velocity,
            mean_velocity_deg_s=mean_velocity,
            sample_count=sample_count,
        )
        self._events.append(event)
        return event


class SaccadeVelocityGraph:
    """Live velocity trace for one eye, styled to match PupilDilationGraph."""

    def __init__(self, tracker, width=900, height=340):
        self.tracker = tracker
        self.width = int(width)
        self.height = int(height)

    def render(self):
        canvas = np.full(
            (self.height, self.width, 3),
            (247, 247, 245),
            dtype=np.uint8,
        )
        rect = (72, 92, self.width - 24, self.height - 28)
        self._draw_header(canvas)
        self._draw_plot(canvas, rect)
        return canvas

    def _draw_header(self, canvas):
        cv2.putText(
            canvas,
            "Saccade velocity",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )

        events = self.tracker.events
        sample_rate = self.tracker.sample_rate_hz
        if not events:
            status = "No saccades detected yet"
        else:
            latest = events[-1]
            rate = self.tracker.saccade_rate_hz
            status = (
                f"Saccades: {len(events)}"
                + (f" ({rate:.2f}/s)" if rate is not None else "")
                + f" | Last: {latest.amplitude_deg:.1f} deg, "
                f"peak {latest.peak_velocity_deg_s:.0f} deg/s, "
                f"{latest.sample_count} sample(s)"
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

        if sample_rate is not None and sample_rate < 120.0:
            cv2.putText(
                canvas,
                f"{sample_rate:.0f} Hz sampling: peak velocity is "
                "underestimated; amplitude and rate remain usable",
                (18, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (40, 90, 200),
                1,
                cv2.LINE_AA,
            )

    def _value_range(self):
        values = [
            velocity
            for _, velocity in self.tracker.history
            if velocity is not None
        ]
        threshold = self.tracker.velocity_threshold_deg_s
        ceiling = max(values) if values else 0.0
        return 0.0, max(threshold * 2.0, ceiling * 1.15, 10.0)

    def _time_range(self):
        history = self.tracker.history
        span = self.tracker.history_seconds
        if not history:
            return 0.0, span
        latest = history[-1][0]
        earliest = history[0][0]
        if latest - earliest < span:
            return earliest, earliest + span
        return latest - span, latest

    def _draw_plot(self, canvas, rect):
        left, top, right, bottom = rect
        cv2.rectangle(canvas, (left, top), (right, bottom), (255, 255, 255), -1)

        value_min, value_max = self._value_range()
        time_min, time_max = self._time_range()
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(round(bottom - fraction * (bottom - top)))
            cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)

        threshold_y = self._map_value_to_y(
            self.tracker.velocity_threshold_deg_s,
            value_min,
            value_max,
            top,
            bottom,
        )
        cv2.line(
            canvas,
            (left, threshold_y),
            (right, threshold_y),
            (55, 145, 225),
            1,
            cv2.LINE_AA,
        )

        for event in self.tracker.events:
            if event.end_timestamp_s < time_min:
                continue
            x = self._map_time_to_x(
                event.start_timestamp_s, time_min, time_max, left, right
            )
            cv2.line(canvas, (x, top), (x, bottom), (205, 225, 245), 1)

        previous_point = None
        previous_timestamp = None
        for timestamp_s, velocity in self.tracker.history:
            if velocity is None:
                previous_point = None
                previous_timestamp = None
                continue

            x = self._map_time_to_x(timestamp_s, time_min, time_max, left, right)
            y = self._map_value_to_y(velocity, value_min, value_max, top, bottom)
            point = (x, y)
            if (
                previous_point is not None
                and timestamp_s - previous_timestamp <= 0.5
            ):
                cv2.line(canvas, previous_point, point, (185, 70, 145), 2, cv2.LINE_AA)
            previous_point = point
            previous_timestamp = timestamp_s

        cv2.rectangle(canvas, (left, top), (right, bottom), (175, 175, 175), 1)
        cv2.putText(
            canvas,
            "Angular velocity (deg/s)",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

        self._draw_y_label(canvas, value_max, left - 8, top + 4)
        self._draw_y_label(
            canvas, (value_min + value_max) / 2.0, left - 8, (top + bottom) // 2 + 4
        )
        self._draw_y_label(canvas, value_min, left - 8, bottom + 4)

    @staticmethod
    def _map_time_to_x(timestamp_s, time_min, time_max, left, right):
        span = time_max - time_min
        if span <= 0:
            return left
        fraction = min(1.0, max(0.0, (timestamp_s - time_min) / span))
        return int(round(left + fraction * (right - left)))

    @staticmethod
    def _map_value_to_y(value, value_min, value_max, top, bottom):
        span = value_max - value_min
        if span <= 0:
            return bottom
        fraction = min(1.0, max(0.0, (value - value_min) / span))
        return int(round(bottom - fraction * (bottom - top)))

    @staticmethod
    def _draw_y_label(canvas, value, right_x, baseline_y):
        label = f"{value:.0f}"
        (text_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
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
