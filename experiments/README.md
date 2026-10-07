# Experiments

This directory contains experiment-specific artifacts and notes.

## Experiment map

| ID | Experiment | Main script | Question |
|---|---|---|---|
| EXP01 | Baseline CNN | `scripts/train_baseline.py` | How well does a small CNN perform from scratch? |
| EXP02 | Transfer learning | `scripts/train.py` | How much does ImageNet pretraining help? |
| EXP03 | Data augmentation | `scripts/train_augmentation.py` | Does basic augmentation improve the scratch baseline? |
| EXP04 | Training-data size | `scripts/train_data_size.py` | How does performance change with fewer labeled samples? |

Generated histories are stored under the corresponding experiment folders when applicable. Final figures and metrics are stored in `results/`.
