# Active Learning for Medical Image Segmentation

[![CI Pipeline](https://github.com/example/medical-active-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/example/medical-active-learning/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)

A production-grade **Active Learning** framework for medical image segmentation, featuring a live real-time dashboard to monitor training experiments from your browser.

---

## ✨ Features

| Feature | Description |
|---|---|
| **4 AL Strategies** | Random · Uncertainty · Diversity · Bayesian MC-Dropout |
| **Real-Time Dashboard** | Live Dice score, epoch progress, analysis images & chart |
| **Stop / Resume** | Gracefully halt any experiment mid-cycle |
| **Auto Log Files** | Full DEBUG trace written to `results/logs/` per run |
| **Reproducible** | Seeded simulation dataset — no external downloads needed |
| **CI/CD** | GitHub Actions pipeline with pytest + import health checks |

---

## 📁 Project Structure

```
active_learning_project/
├── src/
│   ├── model.py          # Lightweight UNet with MC-Dropout support
│   ├── al_strategies.py  # 4 pluggable query strategies
│   ├── engine.py         # Trainer + AL cycle runner
│   ├── dataset.py        # Simulated retinal vessel dataset
│   ├── logger.py         # Structured logging (file + console)
│   ├── main.py           # Experiment orchestrator
│   └── api.py            # FastAPI backend with /run, /stop, /status
├── dashboard/
│   ├── index.html        # Real-time monitoring UI
│   └── style.css         # Dark glassmorphism design
├── configs/
│   └── default.yaml      # All hyperparameters
├── tests/
│   └── test_model.py     # 22 unit tests
├── results/
│   ├── logs/             # Per-run experiment logs
│   └── visual_analysis/  # Live segmentation previews
├── start_server.py       # 🚀 Web server entry point
└── run.py                # CLI-only entry point
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 2. Launch the web dashboard *(recommended)*

```bash
python start_server.py
```

Open **http://localhost:8000** in your browser, then click **🚀 Start Real-Time Training**.

### 3. CLI-only mode

```bash
python run.py
```

Results and logs are saved to `results/`.

---

## 🧪 Run Tests

```bash
# With pytest (recommended)
pytest tests/ -v

# Or standard unittest
python -m unittest discover tests
```

---

## 📊 Active Learning Strategies

### Random Sampling *(Baseline)*
Uniform random selection from the unlabelled pool. Used as the performance lower bound.

### Uncertainty Sampling
Queries samples with the highest pixel-wise predictive entropy — exploiting model confusion.

### Diversity Sampling
K-Means clustering of bottleneck embeddings. Selects samples closest to cluster centroids to maximise coverage of the image manifold.

### MC-Dropout *(Bayesian)*
Monte Carlo Dropout ($T=10$ stochastic passes) estimates epistemic uncertainty via prediction variance — the most principled strategy.

---

## ⚙️ Configuration

All parameters live in `configs/default.yaml`:

```yaml
active_learning:
  initial_size: 2        # Starting labelled pool
  budget_per_cycle: 2    # Samples added per cycle
  cycles: 4              # AL cycles per strategy
  strategies: ["Random", "Uncertainty", "Diversity"]
```

To add a new strategy:
1. Implement the function in `src/al_strategies.py`
2. Add its name to `configs/default.yaml → strategies`
3. Import and register it in `src/main.py → strategies dict`

---

## 📈 Evaluation

- **Dice Similarity Coefficient (DSC)** — primary segmentation metric
- **Label Efficiency Ratio** — % of labels needed to reach 90% of the full-data baseline

---

## 🛡️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/status` | Live experiment state (JSON) |
| `POST` | `/run`    | Start a new experiment |
| `POST` | `/stop`   | Gracefully halt the running experiment |
