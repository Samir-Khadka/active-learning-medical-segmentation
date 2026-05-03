from typing import List, Callable
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def random_sampling(
    model: nn.Module, 
    unlabeled_loader: DataLoader, 
    unlabeled_indices: List[int], 
    budget: int, 
    device: str = 'cpu'
) -> List[int]:
    """
    Baseline strategy: Random uniform selection.
    """
    budget = min(budget, len(unlabeled_indices))
    if budget == 0:
        return []
    rng = np.random.default_rng()
    return rng.choice(unlabeled_indices, budget, replace=False).tolist()

def uncertainty_sampling(
    model: nn.Module, 
    unlabeled_loader: DataLoader, 
    unlabeled_indices: List[int], 
    budget: int, 
    device: str = 'cpu'
) -> List[int]:
    """
    Query samples with the highest average pixel-wise predictive entropy.
    Focuses on 'exploitation' of model confusion.
    """
    budget = min(budget, len(unlabeled_indices))
    if budget == 0:
        return []
    model.eval()
    uncertainties = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            probs = model(images)
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
            avg_entropy = entropy.mean(dim=(1, 2, 3)) 
            uncertainties.extend(avg_entropy.cpu().numpy())
            
    sorted_idx = np.argsort(uncertainties)[::-1]
    return [unlabeled_indices[i] for i in sorted_idx[:budget]]

def mc_dropout_uncertainty(
    model: nn.Module, 
    unlabeled_loader: DataLoader, 
    unlabeled_indices: List[int], 
    budget: int, 
    n_passes: int = 10, 
    device: str = 'cpu'
) -> List[int]:
    """
    Advanced Strategy: Bayesian Uncertainty via Monte Carlo Dropout.
    Captures epistemic uncertainty by measuring variance across stochastic forward passes.
    """
    budget = min(budget, len(unlabeled_indices))
    if budget == 0:
        return []
    model.train()  # Keep dropout active for stochastic inference
    all_variances = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            preds = []
            for _ in range(n_passes):
                preds.append(model(images).cpu().numpy())
            
            preds = np.stack(preds) # (n_passes, batch, C, H, W)
            # Variance of the predicted probabilities
            variance = np.var(preds, axis=0).mean(axis=(1, 2, 3))
            all_variances.extend(variance)
            
    sorted_idx = np.argsort(all_variances)[::-1]
    return [unlabeled_indices[i] for i in sorted_idx[:budget]]

def diversity_sampling(
    model: nn.Module, 
    unlabeled_loader: DataLoader, 
    unlabeled_indices: List[int], 
    budget: int, 
    device: str = 'cpu'
) -> List[int]:
    """
    Query representative samples via K-Means clustering of bottleneck embeddings.
    Ensures broad 'exploration' of the image manifold.
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            emb = model.get_embeddings(images)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    if budget >= len(unlabeled_indices):
        return unlabeled_indices
        
    kmeans = KMeans(n_clusters=budget, n_init=10, random_state=42)
    kmeans.fit(embeddings)
    
    # Selection: Samples closest to cluster centroids
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    
    selected_indices = [unlabeled_indices[i] for i in closest]
    
    # Fallback for small pools or clustering overlap
    if len(selected_indices) < budget:
        remaining = list(set(unlabeled_indices) - set(selected_indices))
        fill_count = budget - len(selected_indices)
        selected_indices.extend(np.random.choice(remaining, fill_count, replace=False).tolist())
        
    return selected_indices
