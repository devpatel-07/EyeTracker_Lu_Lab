import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from train_pupil_unet import (
        PupilDataset,
        TinyUNet,
        combined_loss,
        run_epoch,
    )
except ImportError as exc:
    PupilDataset = None
    TinyUNet = None
    combined_loss = None
    run_epoch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class TrainingComponentTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"train_pupil_unet is not implemented: {IMPORT_ERROR}")

    def test_dataset_returns_normalized_one_channel_tensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "masks").mkdir()

            image = np.full((24, 32), 128, dtype=np.uint8)
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[8:16, 10:22] = 255
            cv2.imwrite(str(root / "images" / "eye.jpg"), image)
            cv2.imwrite(str(root / "masks" / "eye.png"), mask)

            dataset = PupilDataset(
                [{"image": "images/eye.jpg", "mask": "masks/eye.png"}],
                root=root,
                augment=False,
            )
            image_tensor, mask_tensor = dataset[0]

            self.assertEqual(tuple(image_tensor.shape), (1, 24, 32))
            self.assertEqual(tuple(mask_tensor.shape), (1, 24, 32))
            self.assertAlmostEqual(image_tensor.mean().item(), 128 / 255, places=5)
            self.assertEqual(set(torch.unique(mask_tensor).tolist()), {0.0, 1.0})

    def test_unet_preserves_spatial_dimensions(self):
        model = TinyUNet(base_channels=4)
        inputs = torch.rand(2, 1, 32, 64)

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 1, 32, 64))

    def test_combined_loss_is_lower_for_correct_logits(self):
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 2:6, 2:6] = 1
        correct_logits = torch.where(target == 1, 8.0, -8.0)
        wrong_logits = -correct_logits

        correct_loss = combined_loss(correct_logits, target)
        wrong_loss = combined_loss(wrong_logits, target)

        self.assertTrue(torch.isfinite(correct_loss))
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_training_epoch_updates_model_parameters(self):
        torch.manual_seed(3)
        model = TinyUNet(base_channels=4)
        images = torch.rand(2, 1, 32, 32)
        masks = torch.zeros(2, 1, 32, 32)
        masks[:, :, 10:22, 10:22] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.output.weight.detach().clone()

        metrics = run_epoch(model, loader, torch.device("cpu"), optimizer)

        self.assertFalse(torch.equal(before, model.output.weight.detach()))
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["dice"], 0.0)
        self.assertLessEqual(metrics["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
