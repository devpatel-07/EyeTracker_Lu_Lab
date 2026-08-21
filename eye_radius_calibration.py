import math

import numpy as np


class EyeRadiusCalibrator:
    def __init__(
        self,
        target_samples=300,
        percentile=95,
        min_radius_px=10,
        max_radius_px=float("inf"),
    ):
        self.target_samples = target_samples
        self.percentile = percentile
        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px
        self._samples = []
        self._radius_px = None

    @property
    def sample_count(self):
        return len(self._samples)

    @property
    def progress(self):
        return min(1.0, self.sample_count / self.target_samples)

    @property
    def ready(self):
        return self._radius_px is not None

    @property
    def radius_px(self):
        return self._radius_px

    def add(self, value):
        if self.ready or value is None:
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        if not math.isfinite(value):
            return False
        if value < self.min_radius_px or value > self.max_radius_px:
            return False

        self._samples.append(value)
        if self.sample_count >= self.target_samples:
            self._radius_px = float(
                np.percentile(self._samples, self.percentile)
            )
        return True
