import os
import threading
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Callable

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

def load_config(config_path: str = "configs/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Colour palette per strategy for consistent chart look
_STRATEGY_COLORS = {
    'Random':      '#2196F3',
    'Uncertainty': '#FF9800',
    'Diversity':   '#4CAF50',
    'MC_Dropout':  '#F44336',
}

def plot_results(all_histories, full_data_dice, save_path):
    """Render a publication-quality comparison chart and save to disk."""
    if not all_histories:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    for name, history in all_histories.items():
        if not history:
            continue
        x = [h[0] for h in history]
        y = [h[1] for h in history]
        colour = _STRATEGY_COLORS.get(name, '#FFFFFF')
        ax.plot(x, y, marker='o', label=name, linewidth=2.5,
                color=colour, markersize=7, markerfacecolor='white',
                markeredgecolor=colour, markeredgewidth=2)

    # Full-data baseline
    ax.axhline(y=full_data_dice, color='#EF4444', linestyle='--',
               linewidth=2, label=f'Full Data Baseline (100%)', alpha=0.85)
    # 90% target
    ax.axhline(y=full_data_dice * 0.9, color='#10B981', linestyle=':',
               linewidth=1.5, label='90% Target Goal', alpha=0.7)

    ax.set_title('Active Learning Performance: Comparative Analysis',
                 color='white', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Number of Labeled Samples', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Dice Coefficient (Test Set)', color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.grid(True, color='#1e293b', linestyle='-', linewidth=1)
    legend = ax.legend(facecolor='#1e293b', edgecolor='#334155',
                       labelcolor='white', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

def generate_failure_analysis(model, test_dataset, save_dir, device='cpu', progress_callback=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    # Save a montage or a single key analysis image
    idx = 0
    img, mask = test_dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze().numpy()
    
    save_path = os.path.join(save_dir, f'live_analysis.png')
    plt.imsave(save_path, pred, cmap='magma')
    
    if progress_callback:
        progress_callback({"analysis_image_path": f"results/visual_analysis/live_analysis.png?t={int(torch.randint(0, 1000000, (1,)).item())}"})

def main_with_callback(progress_callback: Optional[Callable] = None, stop_event: Optional[threading.Event] = None):
    # Ensure log files are written for every run (deduplication guard is in setup_logging)
    setup_logging()
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, test_dataset = get_dataloaders(train_size=config['data']['train_size'], test_size=config['data']['test_size'])
    
    if progress_callback: 
        progress_callback({
            "current_strategy": "Full Data Baseline",
            "log": "Initializing Full Data Baseline Training..."
        })
    
    full_model = UNet().to(device)
    
    def on_epoch_end_callback(epoch, model):
        generate_failure_analysis(model, test_dataset, os.path.join(config['results']['output_dir'], 'visual_analysis'), device, progress_callback)

    trainer = ActiveLearningTrainer(full_model, device=device, lr=config['model']['lr'], progress_callback=progress_callback)
    trainer.fit(train_dataset, epochs=config['model']['epochs_baseline'], stop_event=stop_event, on_epoch_end=on_epoch_end_callback)
    full_data_dice = trainer.evaluate(test_dataset)
    
    generate_failure_analysis(full_model, test_dataset, os.path.join(config['results']['output_dir'], 'visual_analysis'), device, progress_callback)
    
    all_histories = {}
    al_config = {
        'initial_size': config['active_learning']['initial_size'],
        'budget_per_cycle': config['active_learning']['budget_per_cycle'],
        'cycles': config['active_learning']['cycles'],
        'lr': config['model']['lr'],
        'epochs_al': config['model']['epochs_al'],
        'batch_size': config['data'].get('batch_size', 4)
    }
    
    strategies = {'Random': random_sampling, 'Uncertainty': uncertainty_sampling, 'Diversity': diversity_sampling, 'MC_Dropout': mc_dropout_uncertainty}
    
    for name, strategy_fn in strategies.items():
        if stop_event and stop_event.is_set():
            break
            
        if name in config['active_learning']['strategies'] or name == 'MC_Dropout':
            all_histories[name] = run_active_learning_cycle(
                name, strategy_fn, train_dataset, test_dataset, al_config, 
                device=device, progress_callback=progress_callback, stop_event=stop_event,
                on_epoch_end=on_epoch_end_callback
            )
            # Update plot in real-time
            plot_results(all_histories, full_data_dice, os.path.join(config['results']['output_dir'], 'al_comparison_standard.png'))
            # Generate live analysis for current best model
            generate_failure_analysis(full_model, test_dataset, os.path.join(config['results']['output_dir'], 'visual_analysis'), device, progress_callback)
    
    # Always produce a final plot even if stopped mid-way
    if all_histories:
        plot_results(all_histories, full_data_dice, os.path.join(config['results']['output_dir'], 'al_comparison_standard.png'))
        if progress_callback:
            progress_callback({"progress_percent": 100, "log": "Final results saved."})

def main():
    main_with_callback(None)

if __name__ == "__main__":
    main()
