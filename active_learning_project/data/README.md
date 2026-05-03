# Data Directory

This directory is reserved for the medical imaging datasets used in this framework.

## Recommended Structure
```text
data/
├── raw/
│   └── DRIVE/          # Original DRIVE dataset files
├── processed/
│   └── retinal_sim/    # Cached simulated datasets
└── split_manifests/    # JSON/CSV files defining train/test splits
```

## How to use real data
To transition from the simulation to the official DRIVE dataset:
1. Download the dataset from [Grand Challenge](https://drive.grand-challenge.org/Download/).
2. Place the images in `data/raw/DRIVE/training/images/` and masks in `data/raw/DRIVE/training/1st_manual/`.
3. Update the `data` section in `configs/default.yaml` to point to these paths.
