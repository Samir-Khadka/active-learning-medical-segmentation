# System Architecture & Design Rationale

This document outlines the architectural decisions behind the Active Learning (AL) framework for Medical Image Segmentation.

## 1. Modular U-Net Design
We implemented a parameterizable U-Net (`src/model.py`) that decouples the depth and feature scale from the core logic. This allows for:
- **Dropout Injection**: Essential for Bayesian uncertainty estimation (MC Dropout).
- **Bottleneck Access**: Exposing latent features for diversity-based querying without modifying the forward pass.

## 2. Strategy Pattern for Active Learning
Querying strategies are implemented as functional modules in `src/al_strategies.py`. This follows the **Open-Closed Principle**:
- **Extension**: New strategies (e.g., Coresets, BatchBALD) can be added by implementing a new function without changing the `ActiveLearningTrainer`.
- **Interchangeability**: Strategies can be swapped at runtime via the configuration file.

## 3. Bayesian Uncertainty Estimation
The framework prioritizes **Epistemic Uncertainty**. By using Monte Carlo Dropout, we approximate the posterior predictive distribution $p(y | x, D)$. This is crucial in medical contexts to differentiate between *noise* (aleatoric) and *lack of data* (epistemic).

## 4. Configuration Management
Hyperparameters are decoupled from the code using YAML. This enables **Experiment Tracking** and **Version Control of Research State**.

## 5. Persistence & Auditing
The `results/` directory is structured to persist both high-level results (`.png`) and low-level execution logs (`.log`). This ensures that every scientific claim made by the framework is backed by a non-repudiable audit trail.
