# Project Review — Deep Learning Project 2

This review records the engineering/documentation cleanup applied to the submitted project.

## Fixed

- Rewrote `README.md` so dataset setup, experiments, evaluation, inference, results, and limitations match the implementation.
- Reworked `README_TRANSFER_LEARNING.md` to describe the actual frozen and partial-fine-tuning experiments.
- Corrected stale transfer-learning comparison values:
  - Frozen: 50.92% best validation accuracy
  - Partial fine-tuning: 53.92% best validation accuracy
  - Final test accuracy: 50.88% and 55.26%, respectively.
- Added `best_val_accuracy` to saved transfer-learning histories.
- Fixed `scripts/plot_results.py`, which previously expected a missing `best_val_accuracy` field.
- Removed duplicate checkpoint saving in `scripts/train_augmentation.py`.
- Made `train_augmentation.py` use `configs/augmentation.yaml`.
- Wired `scripts/train.py` to the frozen/partial YAML configs while retaining command-line overrides.
- Updated `configs/frozen.yaml` and `configs/finetune.yaml` to match the actual project strategy and paths.
- Fixed dataset archive extraction so the documented Python 3.11+ support works without relying on Python 3.12-only `filter="data"` syntax.
- Added explicit archive path/link validation before extraction.
- Converted `requirements-lock.txt` from UTF-16 to normal UTF-8 text.
- Updated the baseline pipeline test so it works both as a direct script and as a test module.

## Validation performed

- Python `compileall`: passed.
- Baseline model/training pipeline test: passed.
- CUB limited-data sampling tests: 4/4 passed.
- Transfer-learning figure regeneration: passed.
- `scripts.train --help`: passed.

## Known experimental limitations

The project intentionally remains an experimental study rather than a state-of-the-art CUB benchmark. Most runs use a single seed, and the frozen and partial fine-tuning experiments use different schedules. Therefore, reported differences should not be interpreted as statistically conclusive causal effects.
