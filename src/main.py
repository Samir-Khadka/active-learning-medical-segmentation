import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

from logger import setup_logging
from dataset import get_dataloaders
from model import UNet
from al_strategies import (
    random_sampling, 
    uncertainty_sampling, 
    diversity_sampling, 
    mc_dropout_uncertainty
)
from engine import (
    ActiveLearningTrainer, 
    run_active_learning_cycle
)
# Actually, I'll update main to use the Trainer directly.

def load_config(config_path: str = "configs/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_results(all_histories, full_data_dice, save_path):
    plt.figure(figsize=(10, 6))
    for name, history in all_histories.items():
        x = [h[0] for h in history]
        y = [h[1] for h in history]
        plt.plot(x, y, marker='o', label=name, linewidth=2)
        
    plt.axhline(y=full_data_dice, color='r', linestyle='--', label='Full Data Baseline (100%)', alpha=0.7)
    plt.axhline(y=0.9 * full_data_dice, color='g', linestyle=':', label='90% Target Goal', alpha=0.7)
    
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Dice Coefficient (Test Set)')
    plt.title('Active Learning Performance: Comparative Analysis')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_failure_analysis(model, test_dataset, save_dir, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    indices = [0, 1, 2] 
    for idx in indices:
        img, mask = test_dataset[idx]
        img_in = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_in).cpu().squeeze().numpy()
            
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'analysis_{idx}.png'))
        plt.close()

def main():
    # 1. Initialization
    logger = setup_logging()
    logger.info("Initializing Professional Active Learning Pipeline...")
    
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Target Hardware: {device.upper()}")
    
    # 2. Data Preparation
    train_dataset, test_dataset = get_dataloaders(
        train_size=config['data']['train_size'], 
        test_size=config['data']['test_size']
    )
    logger.info(f"Dataset Loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
    
    # 3. Baseline: Full Data Performance
    logger.info("Computing Full Data Baseline (Upper Bound)...")
    full_model = UNet().to(device)
    trainer = ActiveLearningTrainer(full_model, device=device, lr=config['model']['lr'])
    trainer.fit(train_dataset, epochs=config['model']['epochs_baseline'])
    full_data_dice = trainer.evaluate(test_dataset)
    logger.info(f"Baseline Achievement: {full_data_dice:.4f} Dice")
    
    # 4. Comparative Active Learning Experiments
    all_histories = {}
    al_config = {
        'initial_size': config['active_learning']['initial_size'],
        'budget_per_cycle': config['active_learning']['budget_per_cycle'],
        'cycles': config['active_learning']['cycles'],
        'lr': config['model']['lr'],
        'epochs_al': config['model']['epochs_al'],
        'batch_size': config['data'].get('batch_size', 4)
    }
    
    strategies = {
        'Random': random_sampling,
        'Uncertainty': uncertainty_sampling,
        'Diversity': diversity_sampling,
        'MC_Dropout': mc_dropout_uncertainty
    }
    
    for name, strategy_fn in strategies.items():
        if name in config['active_learning']['strategies'] or name == 'MC_Dropout': # Force MC for demo
            all_histories[name] = run_active_learning_cycle(
                name, strategy_fn, train_dataset, test_dataset, al_config, device=device
            )
    
    # 5. Reporting & Analysis
    logger.info("Experiment cycles complete. Generating final reports...")
    output_plot = os.path.join(config['results']['output_dir'], 'al_comparison_standard.png')
    plot_results(all_histories, full_data_dice, output_plot)
    
    generate_failure_analysis(full_model, test_dataset, os.path.join(config['results']['output_dir'], 'visual_analysis'), device)
    
    # Claim Validation
    final_unc_dice = all_histories.get('Uncertainty', [(0,0)])[-1][1]
    ratio = final_unc_dice / full_data_dice if full_data_dice > 0 else 0
    logger.info(f"Final Comparative Ratio (Uncertainty vs Full): {ratio:.2%}")
    
    if ratio >= 0.90:
        logger.info("[SUCCESS] Active Learning claim validated: 90% performance reached.")
    else:
        logger.warning("[NOTICE] Performance gap remains. Consider increasing training epochs or data diversity.")

if __name__ == "__main__":
    main()
