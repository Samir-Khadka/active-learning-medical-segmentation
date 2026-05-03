import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

logger = logging.getLogger("ActiveLearning")

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return bce_loss + dice_loss

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    num = pred.size(0)
    p_flat = pred.view(num, -1)
    t_flat = target.view(num, -1)
    intersection = (p_flat * t_flat).sum()
    return (2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)

class ActiveLearningTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: str = "cpu", 
        lr: float = 1e-3, 
        batch_size: int = 4,
        progress_callback: Optional[Callable] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = DiceBCELoss()
        self.progress_callback = progress_callback

    def fit(self, dataset: Subset, epochs: int = 20, stop_event: Optional[Any] = None, on_epoch_end: Optional[Callable] = None):
        if len(dataset) == 0: return self.model
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                break
            for images, masks in loader:
                if stop_event and stop_event.is_set():
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            if stop_event and stop_event.is_set():
                break  # Don't fire on_epoch_end for a halted epoch

            if self.progress_callback:
                self.progress_callback({
                    "current_epoch": epoch + 1,
                    "log": f"Epoch [{epoch+1}/{epochs}] training complete."
                })
            
            if on_epoch_end:
                on_epoch_end(epoch + 1, self.model)

    def evaluate(self, dataset: Any) -> float:
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
        return float(np.mean(scores))

def run_active_learning_cycle(
    strategy_name: str,
    strategy_fn: Any,
    train_dataset: Any,
    test_dataset: Any,
    config: Dict[str, Any],
    device: str = "cpu",
    progress_callback: Optional[Callable] = None,
    stop_event: Optional[Any] = None,
    on_epoch_end: Optional[Callable] = None
) -> List[Tuple[int, float]]:
    from model import UNet
    train_indices = list(range(len(train_dataset)))
    rng = np.random.default_rng(42)
    labeled_indices = rng.choice(train_indices, config['initial_size'], replace=False).tolist()
    unlabeled_indices = list(set(train_indices) - set(labeled_indices))
    
    history = []
    total_steps = (config['cycles'] + 1)
    
    for cycle in range(config['cycles'] + 1):
        if progress_callback:
            progress_callback({
                "current_strategy": strategy_name,
                "current_cycle": cycle,
                "progress_percent": int((cycle / total_steps) * 100),
                "log": f"Starting Cycle {cycle} for strategy: {strategy_name}"
            })

        model = UNet().to(device)
        trainer = ActiveLearningTrainer(model, device=device, lr=config['lr'], progress_callback=progress_callback)
        trainer.fit(Subset(train_dataset, labeled_indices), epochs=config['epochs_al'], stop_event=stop_event, on_epoch_end=on_epoch_end)
        
        if stop_event and stop_event.is_set():
            break
            
        score = trainer.evaluate(test_dataset)
        
        if progress_callback:
            progress_callback({
                "latest_dice": round(score, 4),
                "log": f"Cycle {cycle} complete. Score: {score:.4f}"
            })
            
        history.append((len(labeled_indices), score))
        if cycle < config['cycles']:
            unlabeled_subset = Subset(train_dataset, unlabeled_indices)
            loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], shuffle=False)
            new_indices = strategy_fn(model, loader, unlabeled_indices, config['budget_per_cycle'], device=device)
            labeled_indices.extend(new_indices)
            unlabeled_indices = list(set(unlabeled_indices) - set(new_indices))
    return history
