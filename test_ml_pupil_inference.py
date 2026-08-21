import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from train_pupil_unet import TinyUNet

try:
    from ml_pupil_inference import MLPupilDetector
except ImportError as exc:
    MLPupilDetector = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FixedLogitModel(nn.Module):
    def __init__(self, mask):
        super().__init__()
        logits = np.where(mask > 0, 10.0, -10.0).astype(np.float32)
        self.register_buffer("fixed_logits", torch.from_numpy(logits)[None, None])

    def forward(self, inputs):
        return self.fixed_logits.repeat(inputs.shape[0], 1, 1, 1)


class MLPupilDetectorTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"ml_pupil_inference is not implemented: {IMPORT_ERROR}")

    def test_predicted_mask_becomes_full_frame_ellipse(self):
        model_mask = np.zeros((192, 320), dtype=np.uint8)
        cv2.ellipse(model_mask, (160, 96), (40, 24), 15, 0, 360, 255, -1)
        detector = MLPupilDetector(
            FixedLogitModel(model_mask),
            device=torch.device("cpu"),
            input_size=(320, 192),
            threshold=0.5,
        )
        frame = np.full((180, 260, 3), 128, dtype=np.uint8)
        rect = {"x": 30, "y": 20, "w": 200, "h": 120}

        display, ellipse, blink = detector.detect(frame, rect, debug=False)

        self.assertFalse(blink)
        self.assertEqual(display.shape, frame.shape)
        self.assertAlmostEqual(ellipse[0][0], 130, delta=2)
        self.assertAlmostEqual(ellipse[0][1], 80, delta=2)
        self.assertAlmostEqual(max(ellipse[1]), 50, delta=3)

    def test_empty_prediction_is_reported_as_blink(self):
        model_mask = np.zeros((192, 320), dtype=np.uint8)
        detector = MLPupilDetector(
            FixedLogitModel(model_mask),
            device=torch.device("cpu"),
            input_size=(320, 192),
            threshold=0.5,
        )
        frame = np.full((180, 260, 3), 128, dtype=np.uint8)
        rect = {"x": 30, "y": 20, "w": 200, "h": 120}

        _, ellipse, blink = detector.detect(frame, rect, debug=False)

        self.assertTrue(blink)
        self.assertEqual(ellipse, 0)

    def test_detailed_detection_reports_confidence(self):
        model_mask = np.zeros((192, 320), dtype=np.uint8)
        cv2.ellipse(model_mask, (160, 96), (40, 24), 15, 0, 360, 255, -1)
        detector = MLPupilDetector(
            FixedLogitModel(model_mask),
            device=torch.device("cpu"),
            input_size=(320, 192),
            threshold=0.5,
        )
        frame = np.full((180, 260, 3), 128, dtype=np.uint8)
        rect = {"x": 30, "y": 20, "w": 200, "h": 120}

        _, _, blink, confidence = detector.detect_with_confidence(
            frame, rect, debug=False
        )

        self.assertFalse(blink)
        self.assertGreaterEqual(confidence, 0.95)
        self.assertLessEqual(confidence, 1.0)

    def test_detailed_blink_has_zero_confidence(self):
        model_mask = np.zeros((192, 320), dtype=np.uint8)
        detector = MLPupilDetector(
            FixedLogitModel(model_mask),
            device=torch.device("cpu"),
            input_size=(320, 192),
            threshold=0.5,
        )
        frame = np.full((180, 260, 3), 128, dtype=np.uint8)
        rect = {"x": 30, "y": 20, "w": 200, "h": 120}

        _, ellipse, blink, confidence = detector.detect_with_confidence(
            frame, rect, debug=False
        )

        self.assertTrue(blink)
        self.assertEqual(ellipse, 0)
        self.assertEqual(confidence, 0.0)

    def test_nondebug_display_keeps_configured_blue_rect_outline(self):
        detector = MLPupilDetector(
            FixedLogitModel(np.zeros((192, 320), dtype=np.uint8)),
            device=torch.device("cpu"),
            input_size=(320, 192),
            threshold=0.5,
        )
        frame = np.full((180, 260, 3), 128, dtype=np.uint8)
        rect = {"x": 30, "y": 20, "w": 200, "h": 120}

        display, _, blink, _ = detector.detect_with_confidence(
            frame,
            rect,
            debug=False,
        )

        self.assertTrue(blink)
        np.testing.assert_array_equal(display[20, 30], (255, 0, 0))
        np.testing.assert_array_equal(display[0, 0], frame[0, 0])

    def test_checkpoint_loader_restores_model_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            model = TinyUNet(base_channels=4)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "base_channels": 4,
                        "input_width": 320,
                        "input_height": 192,
                    },
                },
                checkpoint_path,
            )

            detector = MLPupilDetector.from_checkpoint(
                checkpoint_path, device="cpu"
            )

            self.assertEqual(detector.input_size, (320, 192))
            self.assertEqual(detector.device, torch.device("cpu"))
            self.assertFalse(detector.model.training)


if __name__ == "__main__":
    unittest.main()
