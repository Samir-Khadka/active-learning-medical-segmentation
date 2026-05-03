import sys
import os
import torch
import unittest

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import UNet
from engine import DiceBCELoss

class TestModelArchitecture(unittest.TestCase):
    """
    Standard regression tests for the core architecture.
    """
    def setUp(self):
        self.device = 'cpu'
        self.model = UNet(n_channels=1, n_classes=1).to(self.device)

    def test_output_shape(self):
        """Verify that output dimensions match input dimensions (Segmentation property)."""
        x = torch.randn(1, 1, 256, 256)
        y = self.model(x)
        self.assertEqual(y.shape, (1, 1, 256, 256))

    def test_embedding_shape(self):
        """Verify the bottleneck feature extractor for diversity sampling."""
        x = torch.randn(1, 1, 256, 256)
        emb = self.model.get_embeddings(x)
        # Expected bottleneck features: 256 channels (based on our refactored model)
        self.assertEqual(emb.shape[1], 256)

    def test_loss_function(self):
        """Verify the hybrid loss range and gradient flow."""
        criterion = DiceBCELoss()
        pred = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = criterion(pred, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

if __name__ == "__main__":
    unittest.main()
