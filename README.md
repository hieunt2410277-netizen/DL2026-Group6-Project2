# DL2026 Group 6 — Project 2
## Fine-Grained Image Recognition under Limited Training Data

A Deep Learning course project studying **fine-grained image classification on CUB-200-2011** under limited labeled-data conditions.

The project compares a small CNN trained from scratch with ImageNet-pretrained ResNet18, and investigates the effect of training-set size and basic data augmentation.

> **Best evaluated configuration:** ResNet18 with partial fine-tuning (`layer4 + fc`) — **55.26% test accuracy** on the official CUB-200-2011 test set.

---

## 1. Research Questions

**RQ1 — Training-data size**  
How does reducing the number of labeled training images affect fine-grained classification performance?

**RQ2 — Transfer learning**  
How effective is ImageNet-pretrained ResNet18 when labeled training data is limited?

**RQ3 — Data augmentation**  
Does basic image augmentation improve generalization for the scratch BaselineCNN?

---

## 2. Dataset

The project uses **CUB-200-2011 (Caltech-UCSD Birds-200-2011)**:

- 200 bird species
- 11,788 images
- 5,994 official training images
- 5,794 official test images

Official source: <https://data.caltech.edu/records/65de6-vp158>

The official test split is never used for training. The 5,994 official training images are split **per class** into:

| Split | Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Official test | 5,794 |

The split uses **seed 42**. The split is performed per class to preserve class balance. See [`DATA.md`](DATA.md) for the complete data protocol.

---

## 3. Models and Experiments

### EXP01 — BaselineCNN

A lightweight CNN trained from scratch:

- 3 convolution blocks
- Batch normalization
- ReLU
- max pooling
- adaptive average pooling
- dropout
- 200-class linear classifier

It is a controlled reference model, **not a state-of-the-art CUB model**.

### EXP02 — Transfer Learning

**Frozen ResNet18**

- ImageNet-pretrained ResNet18
- pretrained backbone frozen
- only the final classifier is trained
- 10 epochs, learning rate `1e-3`
- best-validation checkpoint is used for final evaluation

**Partial fine-tuning**

- `conv1` through `layer3` frozen
- `layer4` and final `fc` trainable
- 3 epochs, learning rate `1e-4`
- weight decay `1e-4`
- frozen features are cached to reduce CPU training time
- best-validation checkpoint is used for final evaluation

### EXP03 — Basic Augmentation

BaselineCNN trained on the full default training split with:

- Random resized crop
- Random horizontal flip

The validation and test transformations remain deterministic.

### EXP04 — Training-data size

BaselineCNN trained with:

- 5 images/class → 1,000 training images
- 10 images/class → 2,000
- 15 images/class → 3,000
- 20 images/class → 4,000

The validation and official test sets remain fixed.

---

## 4. Final Results

All values below match the final report and the saved evaluation artifacts in `results/final_evaluation/`.

| Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| BaselineCNN | 4.42% | 3.80% | 4.52% | 3.17% |
| ResNet18 Frozen | 50.88% | 55.40% | 51.23% | 51.40% |
| **ResNet18 Partial Fine-Tuning** | **55.26%** | **58.84%** | **55.49%** | **54.47%** |
| BaselineCNN + Augmentation | 4.57% | 5.02% | 4.68% | 3.36% |
| 5 images/class | 1.85% | 1.72% | 1.94% | 1.18% |
| 10 images/class | 3.07% | 2.81% | 3.12% | 2.10% |
| 15 images/class | 3.38% | 2.55% | 3.44% | 2.37% |
| 20 images/class | 4.90% | 4.16% | 4.95% | 3.67% |

### Interpretation

- Increasing the available training data produced an upward accuracy trend for the scratch baseline.
- Transfer learning was the strongest factor in these experiments.
- Partial fine-tuning outperformed the frozen ResNet18 configuration by **4.38 percentage points** in test accuracy.
- Basic augmentation improved the scratch baseline by only **0.15 percentage points**.
- The transfer-learning configurations were trained with different schedules, so the frozen-vs-partial difference should be treated as an **observed experimental result**, not a controlled causal estimate.
- BaselineCNN and ResNet18 also differ in architecture, so their complete performance gap cannot be attributed to pretraining alone.
- Experiments were generally run once with seed 42; no mean ± standard deviation is reported.

---

## 5. Project Structure

```text
DL2026-Group6-Project2/
├── configs/                 # Experiment configuration files
├── data/                    # Dataset location (images not committed)
│   ├── raw/
│   └── processed/
├── experiments/             # Experiment-specific notes/artifacts
├── report/                  # Final report and comparison notes
├── results/
│   ├── figures/             # Training curves and comparison plots
│   ├── final_evaluation/   # Final metrics + confusion matrices
│   └── metrics/             # Training histories
├── scripts/
│   ├── train_baseline.py
│   ├── train.py
│   ├── train_augmentation.py
│   ├── train_data_size.py
│   ├── predict.py
│   └── export_model.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── checkpoints/             # Model checkpoints (not committed)
├── DATA.md
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## 6. Installation

### Requirements

- Python **3.11+**
- PyTorch
- torchvision
- CPU or CUDA-capable GPU

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the recorded environment snapshot:

```powershell
python -m pip install -r requirements-lock.txt
```

Verify:

```powershell
python -c "import torch, torchvision, sklearn, PIL, yaml; print('Dependencies OK')"
```

> `requirements-lock.txt` is an environment snapshot. If installing on a different platform or CUDA setup, the unpinned `requirements.txt` may be more appropriate.

---

## 7. Download Dataset and Checkpoint

### 7.1 CUB-200-2011 Dataset

Download CUB-200-2011 from the official source:

<https://data.caltech.edu/records/65de6-vp158>

The dataset is required for training, evaluation, and full experiment reproduction. The
image data is intentionally **not committed to the repository**.

Either extract it to:

```text
data/CUB_200_2011/
```

or place the original archive here:

```text
data/CUB_200_2011.tgz
```

The project can extract the archive automatically.

Verify the dataset:

```powershell
python -m src.data.dataset
```

Expected:

```text
Number of classes: 200
Train samples: 4794
Validation samples: 1200
Test samples: 5794
Image batch shape: torch.Size([32, 3, 224, 224])
```

### 7.2 Pre-trained Checkpoint

If you only want to run the inference demo, you **do not need to download the CUB-200-2011
dataset**. Download the exported full ResNet18 checkpoint from the **GitHub Release** for this
repository:

- **File:** `resnet18_partial_full.pth`
- **Size:** approximately 43.1 MB
- **Purpose:** inference with the best partial fine-tuning model

> The checkpoint is distributed through GitHub Releases rather than committed to the repository
> because it is a relatively large binary file.

After downloading, place it at:

```text
checkpoints/resnet18_partial_full.pth
```

The `checkpoints/` directory is intentionally ignored by Git except for its `.gitkeep` file.
See [`checkpoints/README.md`](checkpoints/README.md) for the checkpoint workflow.

---

## 8. Quick Inference

This is the fastest way to verify the exported model without preparing the full CUB dataset.

1. Install the Python dependencies from [Section 6](#6-installation).
2. Download `resnet18_partial_full.pth` from the GitHub Release and place it in
   `checkpoints/`.
3. Provide any RGB bird image as input.
4. Run:

```powershell
python -m scripts.predict --image "path/to/bird.jpg" --checkpoint "checkpoints/resnet18_partial_full.pth" --top-k 5
```

The script uses the fixed 200-class list in `configs/classes.txt` and prints the Top-k species
predictions and softmax probabilities. The exported full checkpoint contains the complete
ResNet18 state, so **the CUB dataset is not required for inference**.

> The model is a closed-set classifier trained on the 200 CUB classes. It will always return one
> of those classes, even when the input is a non-bird image or a species outside CUB-200-2011.

---

## 9. Reproduce the Experiments

Run all commands from the repository root.

### EXP01 — Baseline

```powershell
python -m scripts.train_baseline
```

Produces:

```text
checkpoints/baseline_final.pth
results/metrics/baseline.json
results/figures/baseline_accuracy.png
results/figures/baseline_loss.png
```

### EXP02 — Transfer Learning

Frozen backbone:

```powershell
python -m scripts.train --strategy frozen --epochs 10 --batch-size 32
```

Partial fine-tuning:

```powershell
python -m scripts.train --strategy partial_finetune --epochs 3 --batch-size 32
```

Produces:

```text
checkpoints/resnet18_frozen_best.pth
checkpoints/resnet18_partial_finetune_best.pth
results/metrics/frozen.json
results/metrics/partial_finetune.json
```

Regenerate transfer-learning figures:

```powershell
python -m scripts.plot_results
```

Compare the saved histories:

```powershell
python -m scripts.compare_results
```

### EXP03 — Basic augmentation

```powershell
python -m scripts.train_augmentation
```

### EXP04 — Training-data size

```powershell
python -m scripts.train_data_size
```

This runs all four settings: 5, 10, 15, and 20 images/class.

---

## 10. Final Evaluation

The evaluation script reports:

- Accuracy
- Macro precision / recall / F1
- Weighted precision / recall / F1
- Confusion matrix

Examples:

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/baseline_final.pth --name baseline

python "EVAL part/EVAL.py" --model-type resnet18_frozen --checkpoint checkpoints/resnet18_frozen_best.pth --name resnet18_frozen

python "EVAL part/EVAL.py" --model-type resnet18_partial --checkpoint checkpoints/resnet18_partial_finetune_best.pth --name resnet18_partial
```

Augmentation:

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/augmentation_final.pth --name augmentation
```

Limited-data checkpoints:

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_5.pth --name data_size_5
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_10.pth --name data_size_10
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_15.pth --name data_size_15
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_20.pth --name data_size_20
```

Results are saved under:

```text
results/final_evaluation/
```

---

## 11. Detailed Inference and Model Export

The repository includes `scripts/predict.py` for the exported full ResNet18 checkpoint. The
shortest demo workflow is described in [Section 8](#8-quick-inference).

If the project release contains:

```text
resnet18_partial_full.pth
```

place it at:

```text
checkpoints/resnet18_partial_full.pth
```

Then:

```powershell
python -m scripts.predict --image "path/to/bird.jpg" --checkpoint "checkpoints/resnet18_partial_full.pth" --top-k 5
```

The full exported checkpoint contains the complete ResNet18 state, so inference does **not** require the CUB dataset.

The class names are read from:

```text
configs/classes.txt
```

### Exporting the full checkpoint

If you have the partial fine-tuning checkpoint:

```powershell
python -m scripts.export_model
```

This reconstructs the ImageNet-pretrained backbone and combines it with the trained `layer4` and `fc`.

> Exporting from the partial checkpoint may require downloading the standard ImageNet-pretrained ResNet18 weights once.

---

## 12. Tests

Run the automated tests with:

```powershell
python -m pytest tests
```

Or run the baseline pipeline test directly:

```powershell
python tests/test_baseline_pipeline.py
```

Dataset sampling tests:

```powershell
python -m experiments.exp04_data_size.test_dataset_sampling
```

---

## 13. Reproducibility

The project uses seed **42** for dataset splitting and limited-data sampling.

Important limitations:

1. Most experiments were run once, so statistical uncertainty across seeds is not measured.
2. The official test set is kept separate from training and validation.
3. BaselineCNN and ResNet18 differ in architecture as well as initialization.
4. Frozen and partial fine-tuning use different epoch counts and learning rates.
5. Basic augmentation was intentionally simple and produced only a small observed improvement.
6. The classifier is a closed-set 200-class classifier and is not designed to reject unknown species or non-bird images.

For data-specific details, see [`DATA.md`](DATA.md).

---

## 14. Reports and Results

- Final project report: [`report/Group6_Project2_Report.pdf`](report/Group6_Project2_Report.pdf)
- Transfer-learning analysis: [`report/TRANSFER_LEARNING_COMPARISON.md`](report/TRANSFER_LEARNING_COMPARISON.md)
- Evaluation findings: [`EVAL part/evaluation_report.md`](EVAL%20part/evaluation_report.md)
- Dataset documentation: [`DATA.md`](DATA.md)

---

## 15. Team

**Group 6 — Deep Learning 2026, Project 2**

The repository contains the implementation, experiment artifacts, evaluation pipeline, and final report for the project.
