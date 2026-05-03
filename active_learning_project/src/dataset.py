import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import random

class SimulatedVesselDataset(Dataset):
    """
    A simulated dataset that mimics retinal vessel segmentation (DRIVE-like).
    This ensures the project is runnable without external large downloads.
    """
    def __init__(self, num_samples=40, image_size=256, seed=42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.images = []
        self.masks = []
        
        for i in range(num_samples):
            img, mask = self._generate_sample()
            self.images.append(img)
            self.masks.append(mask)
            
    def _generate_sample(self):
        # Create empty canvas
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Add background illumination (retinal glow)
        center = (self.image_size // 2, self.image_size // 2)
        cv2.circle(img, center, self.image_size // 2, 40, -1)
        
        # Add optic disc
        disc_pos = (random.randint(self.image_size//4, 3*self.image_size//4), 
                    random.randint(self.image_size//4, 3*self.image_size//4))
        cv2.circle(img, disc_pos, 20, 180, -1)
        
        # Add vessels (random lines/curves)
        num_vessels = random.randint(15, 25)
        for _ in range(num_vessels):
            start_pt = disc_pos
            for _ in range(3): # Segmented vessel
                end_pt = (start_pt[0] + random.randint(-60, 60), 
                          start_pt[1] + random.randint(-60, 60))
                thickness = random.randint(1, 3)
                cv2.line(mask, start_pt, end_pt, 255, thickness)
                cv2.line(img, start_pt, end_pt, 120, thickness)
                start_pt = end_pt
        
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

def get_dataloaders(batch_size=4, train_size=32, test_size=8):
    total = train_size + test_size
    full_dataset = SimulatedVesselDataset(num_samples=total)
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, test_dataset
