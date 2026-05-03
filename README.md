# Active Learning for Medical Image Segmentation: Architectural Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)

## 📝 Design Philosophy
This repository implements a production-grade Active Learning (AL) framework for medical imaging. Drawing from 50+ years of collective software engineering wisdom, the architecture prioritizes **modularity**, **reproducibility**, and **defensive implementation**.

### Core Engineering Principles:
- **Separation of Concerns**: Decoupled query strategies from the training engine.
- **Config-Driven Orchestration**: All hyperparameters are managed via validated YAML configurations.
- **Auditability**: Comprehensive logging system that captures state transitions, metric drift, and hardware utilization.
- **Bayesian Robustness**: Implementation of Monte Carlo Dropout for reliable uncertainty estimation.

## 📁 System Architecture
```text
active_learning_project/
├── src/
│   ├── model.py          # Modular UNet with Dropout injection
│   ├── al_strategies.py  # Strategy Pattern implementation for AL
│   ├── engine.py         # Encapsulated Trainer and Cycle Runner
│   ├── dataset.py        # High-fidelity Medical Simulation
│   ├── logger.py         # Unified Logging System
│   └── main.py           # Experiment Orchestrator
├── configs/
│   └── default.yaml      # Centralized Configuration
├── results/
│   ├── logs/             # Detailed Audit Trails
│   └── visual_analysis/  # Model Interpretability Assets
├── requirements.txt      # Dependency Management
└── run.py                # System Entry Point
```

## 🚀 Advanced Features
### 1. Epistemic Uncertainty via MC-Dropout
Unlike frequentist models, our framework utilizes stochastic forward passes ($T=10$) to approximate the posterior distribution. This allows for querying samples where the model's internal representations are genuinely unstable, rather than just high-entropy noise.

### 2. High-Fidelity Retinal Simulation
To ensure 100% out-of-the-box reproducibility without multi-gigabyte downloads, we use a vectorized simulation of retinal vasculature, including optic disc modeling and Gaussian artifact noise.

## 🛠️ Operational Instructions
### Prerequisites
Ensure a Python 3.11+ environment is active.
```bash
pip install -r requirements.txt
```

### Execution
The system supports a single-entry point for full experiment replication:
```bash
python run.py
```

## 📊 Evaluation Metrics
- **DSC (Dice Similarity Coefficient)**: The primary structural metric.
- **Label Efficiency Ratio**: The % of labels required to reach 90% of the `Full-Data Baseline` (FDB).

## 🛡️ Maintenance & Scalability
This framework is designed to be extended. To add a new strategy:
1. Implement the strategy function in `src/al_strategies.py`.
2. Register the strategy in the `configs/default.yaml` list.
3. The orchestrator will automatically handle the parallel evaluation and reporting.
