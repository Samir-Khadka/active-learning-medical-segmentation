import sys
import os
import torch
import unittest

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import UNet
from engine import DiceBCELoss, ActiveLearningTrainer, calculate_dice
from dataset import SimulatedVesselDataset, get_dataloaders
from al_strategies import random_sampling, uncertainty_sampling
from torch.utils.data import DataLoader, Subset


class TestModelArchitecture(unittest.TestCase):
    """Regression tests for UNet architecture."""

    def setUp(self):
        self.device = 'cpu'
        self.model = UNet(n_channels=1, n_classes=1).to(self.device)

    def test_output_shape(self):
        """Output spatial dimensions must match input (segmentation invariant)."""
        x = torch.randn(1, 1, 256, 256)
        y = self.model(x)
        self.assertEqual(y.shape, (1, 1, 256, 256))

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        x = torch.randn(2, 1, 256, 256)
        y = self.model(x)
        self.assertTrue(y.min().item() >= 0.0 - 1e-6)
        self.assertTrue(y.max().item() <= 1.0 + 1e-6)

    def test_embedding_shape(self):
        """Bottleneck feature extractor must return correct channel count."""
        x = torch.randn(1, 1, 256, 256)
        emb = self.model.get_embeddings(x)
        self.assertEqual(emb.shape[1], 256)

    def test_embedding_is_2d(self):
        """Embeddings must be (batch, features) — no spatial dims."""
        x = torch.randn(3, 1, 256, 256)
        emb = self.model.get_embeddings(x)
        self.assertEqual(emb.dim(), 2)
        self.assertEqual(emb.shape[0], 3)


class TestLossFunction(unittest.TestCase):
    """Tests for DiceBCELoss."""

    def setUp(self):
        self.criterion = DiceBCELoss()

    def test_loss_is_positive(self):
        pred   = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = self.criterion(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_gradient_flows(self):
        pred   = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0]]]])
        loss = self.criterion(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())

    def test_perfect_prediction_low_loss(self):
        """Near-perfect prediction should yield near-zero loss."""
        eps = 1e-3
        pred   = torch.clamp(torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), eps, 1 - eps)
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = self.criterion(pred, target)
        self.assertLess(loss.item(), 0.5)


class TestDiceMetric(unittest.TestCase):
    """Tests for the standalone Dice calculation."""

    def test_perfect_match(self):
        pred   = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        score  = calculate_dice(pred, target)
        self.assertAlmostEqual(score.item(), 1.0, places=4)

    def test_zero_overlap(self):
        pred   = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        score  = calculate_dice(pred, target)
        self.assertLess(score.item(), 0.01)


class TestDataset(unittest.TestCase):
    """Tests for the simulated vessel dataset."""

    def test_dataset_length(self):
        ds = SimulatedVesselDataset(num_samples=10)
        self.assertEqual(len(ds), 10)

    def test_image_shape(self):
        ds  = SimulatedVesselDataset(num_samples=4, image_size=64)
        img, mask = ds[0]
        self.assertEqual(img.shape,  (1, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))

    def test_pixel_range(self):
        ds  = SimulatedVesselDataset(num_samples=4, image_size=64)
        img, mask = ds[0]
        self.assertTrue(img.min().item()  >= 0.0 - 1e-5)
        self.assertTrue(img.max().item()  <= 1.0 + 1e-5)
        self.assertTrue(mask.min().item() >= 0.0 - 1e-5)
        self.assertTrue(mask.max().item() <= 1.0 + 1e-5)

    def test_get_dataloaders_split(self):
        train_ds, test_ds = get_dataloaders(train_size=8, test_size=4)
        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(test_ds),  4)


class TestStrategies(unittest.TestCase):
    """Tests for active learning query strategies."""

    def setUp(self):
        self.model = UNet(n_channels=1, n_classes=1)
        ds = SimulatedVesselDataset(num_samples=8, image_size=64)
        self.loader = DataLoader(ds, batch_size=4, shuffle=False)
        self.indices = list(range(8))

    def test_random_sampling_count(self):
        selected = random_sampling(self.model, self.loader, self.indices, budget=3)
        self.assertEqual(len(selected), 3)

    def test_random_sampling_subset_of_pool(self):
        selected = random_sampling(self.model, self.loader, self.indices, budget=3)
        for idx in selected:
            self.assertIn(idx, self.indices)

    def test_random_sampling_budget_guard(self):
        """Budget larger than pool should not crash — returns whole pool."""
        selected = random_sampling(self.model, self.loader, self.indices, budget=100)
        self.assertEqual(len(selected), len(self.indices))

    def test_uncertainty_sampling_count(self):
        selected = uncertainty_sampling(self.model, self.loader, self.indices, budget=3)
        self.assertEqual(len(selected), 3)

    def test_uncertainty_budget_guard(self):
        selected = uncertainty_sampling(self.model, self.loader, self.indices, budget=999)
        self.assertEqual(len(selected), len(self.indices))


class TestTrainer(unittest.TestCase):
    """Smoke tests for ActiveLearningTrainer."""

    def test_fit_and_evaluate(self):
        ds    = SimulatedVesselDataset(num_samples=4, image_size=64)
        model = UNet(n_channels=1, n_classes=1)
        trainer = ActiveLearningTrainer(model, device='cpu', lr=1e-3, batch_size=2)
        trainer.fit(ds, epochs=1)
        score = trainer.evaluate(ds)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fit_empty_dataset_no_crash(self):
        """Empty labeled set must not crash."""
        ds    = SimulatedVesselDataset(num_samples=4, image_size=64)
        empty = Subset(ds, [])
        model = UNet(n_channels=1, n_classes=1)
        trainer = ActiveLearningTrainer(model, device='cpu')
        trainer.fit(empty, epochs=2)  # Should return silently


if __name__ == '__main__':
    unittest.main(verbosity=2)
