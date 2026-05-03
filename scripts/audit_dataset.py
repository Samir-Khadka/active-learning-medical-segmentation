import sys
import os
import torch
import numpy as np

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import SimulatedVesselDataset

def main():
    """
    Utility script to verify dataset integrity and display statistical summaries.
    """
    print("--- Dataset Integrity Audit ---")
    dataset = SimulatedVesselDataset(num_samples=20)
    
    images = []
    masks = []
    
    for i in range(len(dataset)):
        img, mask = dataset[i]
        images.append(img.numpy())
        masks.append(mask.numpy())
        
    images = np.stack(images)
    masks = np.stack(masks)
    
    print(f"Total Samples: {len(dataset)}")
    print(f"Image Shape: {images.shape}")
    print(f"Image Intensity: Mean={np.mean(images):.4f}, Std={np.std(images):.4f}")
    print(f"Mask Coverage: {np.mean(masks)*100:.2f}% (Retinal Vasculature Density)")
    print("Audit Complete.")

if __name__ == "__main__":
    main()
