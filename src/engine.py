import logging
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("ActiveLearning")

class DiceBCELoss(nn.Module):
    """
    Hybrid loss function for medical image segmentation.
    Combining BCE for pixel-wise accuracy and Dice for structural overlap.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce_loss = self.bce(inputs, targets)
        
        # Dice component
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return bce_loss + dice_loss

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Computes the Dice Similarity Coefficient (DSC)."""
    num = pred.size(0)
    p_flat = pred.view(num, -1)
    t_flat = target.view(num, -1)
    intersection = (p_flat * t_flat).sum()
    return (2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)

class ActiveLearningTrainer:
    """
    A professional-grade Trainer class to manage model lifecycle, training, and evaluation.
    """
    def __init__(
        self, 
        model: nn.Module, 
        device: str = "cpu", 
        lr: float = 1e-3, 
        batch_size: int = 4
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = DiceBCELoss()

    def fit(self, dataset: Subset, epochs: int = 20) -> nn.Module:
        """Trains the model on a labeled subset."""
        if len(dataset) == 0:
            logger.warning("Attempted to train on an empty dataset.")
            return self.model

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, masks in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.debug(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(loader):.4f}")
                
        return self.model

    def evaluate(self, dataset: Any) -> float:
        """Evaluates the model on a test set using Dice Coefficient."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores = []
        
        with torch.no_grad():
            for images, masks in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                preds = (outputs > 0.5).float()
                dice = calculate_dice(preds, masks)
                scores.append(dice.item())
                
        mean_score = np.mean(scores)
        return float(mean_score)

def run_active_learning_cycle(
    strategy_name: str,
    strategy_fn: Any,
    train_dataset: Any,
    test_dataset: Any,
    config: Dict[str, Any],
    device: str = "cpu"
) -> List[Tuple[int, float]]:
    """
    Executes a multi-cycle active learning experiment.
    """
    from model import UNet
    
    logger.info(f"Starting Active Learning experiment with strategy: {strategy_name}")
    
    train_indices = list(range(len(train_dataset)))
    # Fixed seed for reproducibility - essential for pro experiments
    rng = np.random.default_rng(42)
    labeled_indices = rng.choice(train_indices, config['initial_size'], replace=False).tolist()
    unlabeled_indices = list(set(train_indices) - set(labeled_indices))
    
    history = []
    
    for cycle in range(config['cycles'] + 1):
        # Initialize/Re-initialize model for current pool
        model = UNet().to(device)
        trainer = ActiveLearningTrainer(model, device=device, lr=config['lr'])
        
        # Train on current labeled pool
        labeled_subset = Subset(train_dataset, labeled_indices)
        trainer.fit(labeled_subset, epochs=config['epochs_al'])
        
        # Evaluate
        score = trainer.evaluate(test_dataset)
        logger.info(f"Strategy: {strategy_name} | Cycle: {cycle} | Labels: {len(labeled_indices)} | Dice: {score:.4f}")
        history.append((len(labeled_indices), score))
        
        if cycle < config['cycles']:
            # Query more samples
            unlabeled_subset = Subset(train_dataset, unlabeled_indices)
            loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], shuffle=False)
            
            new_indices = strategy_fn(
                model, 
                loader, 
                unlabeled_indices, 
                config['budget_per_cycle'], 
                device=device
            )
            
            labeled_indices.extend(new_indices)
            unlabeled_indices = list(set(unlabeled_indices) - set(new_indices))
            
    return history
