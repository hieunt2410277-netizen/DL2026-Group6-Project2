# DL2026-Group6-Project2

Deep Learning Course Project

**Topic:** Fine-Grained Image Recognition under Limited Training Data  
**Dataset:** CUB-200-2011  
**Group:** 6  
**Project ID:** 2

## Overview

Fine-grained image recognition aims to distinguish between visually similar categories, such as closely related bird species. This project investigates how deep learning models behave when only a limited amount of labeled training data is available.

The project focuses on three factors:

- Training-data size
- Transfer learning
- Data augmentation

A small CNN trained from scratch is used as a controlled reference baseline. ImageNet-pretrained ResNet18 is then evaluated with a frozen backbone and with partial fine-tuning.

## Quick Inference Demo - No CUB Dataset Required

The best-performing evaluated model is **ResNet18 with partial fine-tuning**, with **55.26% test accuracy** on the official CUB-200-2011 test split.

A complete inference checkpoint is available through GitHub Releases:

**Checkpoint:**  
https://github.com/hieunt2410277-netizen/DL2026-Group6-Project2/releases/download/v1.0.0-demo/resnet18_partial_full.pth

Save it as:

```text
checkpoints/resnet18_partial_full.pth
```

Then run:

```powershell
python -m scripts.predict --image "path/to/bird.jpg" --checkpoint "checkpoints/resnet18_partial_full.pth" --top-k 5
```

The inference demo requires only:

- Project dependencies
- `configs/classes.txt`
- The checkpoint above
- One input image

The full CUB-200-2011 dataset is **not required for inference**. The checkpoint stores learned model weights, not training images.

## Research Questions

1. **RQ1 - Training Data Size:** How does reducing the number of labeled training images affect fine-grained classification performance?
2. **RQ2 - Transfer Learning:** How effective is transfer learning when labeled training data is limited?
3. **RQ3 - Data Augmentation:** Does basic data augmentation improve generalization for the baseline CNN?

For RQ2, the project also compares a frozen pretrained backbone with partial fine-tuning.

## Dataset

The project uses the **CUB-200-2011 (Caltech-UCSD Birds-200-2011)** dataset.

- Classes: 200 bird species
- Total images: 11,788
- Official training images: 5,994
- Official test images: 5,794
- Official source: https://data.caltech.edu/records/65de6-vp158

The official training split is divided class-wise into training and validation subsets using seed 42:

| Split | Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Official test | 5,794 |

The official test set is kept separate from model development.

## Models

### 1. BaselineCNN

A small convolutional neural network trained from scratch. It contains three convolution blocks with BatchNorm, ReLU, and max pooling, followed by adaptive average pooling, dropout, and a 200-class linear classifier.

The baseline is intentionally simple and serves as a reference model for studying limited-data behavior. It is not intended to be a state-of-the-art CUB classifier.

### 2. ResNet18 - Frozen Backbone

An ImageNet-pretrained ResNet18 with the pretrained feature extractor frozen. Only the final classification layer is trained.

### 3. ResNet18 - Partial Fine-Tuning

An ImageNet-pretrained ResNet18 where early layers remain frozen while `layer4` and the final classifier are trainable.

### 4. BaselineCNN with Basic Augmentation

The same BaselineCNN trained with random resized crop and random horizontal flip.

## Main Results

All final metrics below were computed on the official CUB-200-2011 test split.

| Model / Setting | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| BaselineCNN | 4.42% | 3.80% | 4.52% | 3.17% |
| ResNet18 Frozen | 50.88% | 55.40% | 51.23% | 51.40% |
| ResNet18 Partial Fine-Tuning | **55.26%** | **58.84%** | **55.49%** | **54.47%** |
| BaselineCNN + Basic Augmentation | 4.57% | 5.02% | 4.68% | 3.36% |
| 5 images/class | 1.85% | 1.72% | 1.94% | 1.18% |
| 10 images/class | 3.07% | 2.81% | 3.12% | 2.10% |
| 15 images/class | 3.38% | 2.55% | 3.44% | 2.37% |
| 20 images/class | 4.90% | 4.16% | 4.95% | 3.67% |

### Interpretation

- **RQ1:** Baseline accuracy increases from 1.85% at 5 images/class to 4.90% at 20 images/class. This is an observed upward trend, not a universal law.
- **RQ2:** The pretrained ResNet18 configurations substantially outperform the scratch BaselineCNN in this project setup. Because architecture and initialization differ between BaselineCNN and ResNet18, the full performance gap cannot be attributed only to pretraining.
- **Frozen vs Partial:** Partial fine-tuning achieves 55.26% versus 50.88% for the frozen strategy. However, the two runs use different training schedules and hyperparameters, so the 4.38 percentage-point difference is reported as an observed result rather than a controlled causal estimate of fine-tuning alone.
- **RQ3:** Basic augmentation changes BaselineCNN accuracy from 4.42% to 4.57%, only +0.15 percentage points. Without repeated runs across multiple seeds, this does not provide strong evidence of a stable augmentation benefit.

## Checkpoint Selection Policy

Checkpoint handling differs by experiment and is documented explicitly:

- **BaselineCNN:** final-epoch checkpoint (`baseline_final.pth`)
- **BaselineCNN + augmentation:** final-epoch checkpoint (`augmentation_final.pth`)
- **Limited-data experiments:** final-epoch checkpoints (`data_size_*.pth`)
- **ResNet18 frozen:** best validation checkpoint (`resnet18_frozen_best.pth`)
- **ResNet18 partial fine-tuning:** best validation checkpoint (`resnet18_partial_finetune_best.pth`)

Validation data is used to monitor training in all experiments, while best-validation checkpoint selection is applied only to the two ResNet18 transfer-learning experiments.

## Experimental Configuration

### Baseline / Augmentation / Limited-Data

- Image size: 224 x 224
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Seed: 42
- Image normalization: ImageNet mean/std

Original CUB labels 1-200 are converted to PyTorch class indices 0-199 in the shared dataset pipeline.

### Transfer Learning

**Frozen ResNet18**
- Epochs: 10
- Learning rate: 1e-3
- Trainable part: final classifier

**Partial Fine-Tuning**
- Epochs: 3
- Learning rate: 1e-4
- Weight decay: 1e-4
- Trainable part: `layer4` + final classifier

These configurations were not designed as a perfectly controlled ablation between frozen and partial fine-tuning. The reported comparison therefore reflects the evaluated configurations.

## Project Structure

```text
DL2026-Group6-Project2/
|
├── configs/
│   └── Experiment configuration files
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── experiments/
│   ├── exp01_baseline/
│   ├── exp02_transfer_learning/
│   ├── exp03_augmentation/
│   └── exp04_data_size/
├── notebooks/
├── report/
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   └── metrics/
├── scripts/
├── slides/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── checkpoints/
├── DATA.md
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/hieunt2410277-netizen/DL2026-Group6-Project2.git
cd DL2026-Group6-Project2
```

### 2. Create a Virtual Environment

Python 3.11 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

For the exact environment snapshot used during final cleanup, see `requirements-lock.txt`.

### 4. Verify Installation

```powershell
python -c "import torch, torchvision, sklearn, PIL, yaml; print('Dependencies OK')"
```

## Dataset Setup for Training and Evaluation

Download CUB-200-2011 from:

https://data.caltech.edu/records/65de6-vp158

Extract it to:

```text
data/CUB_200_2011/
```

Alternatively, place `CUB_200_2011.tgz` inside `data/`. The data pipeline can extract the archive when needed.

For complete data documentation, preprocessing, split logic, and reproducibility information, see [DATA.md](DATA.md).

### Verify Dataset Loading

```powershell
python -m src.data.dataset
```

Expected information:

```text
Number of classes: 200
Train samples: 4794
Validation samples: 1200
Test samples: 5794
Image batch shape: torch.Size([32, 3, 224, 224])
```

## Reproducing Experiments

Run commands from the repository root after installing dependencies and preparing CUB-200-2011.

### EXP01 - Baseline CNN

```powershell
python -m scripts.train_baseline
```

Outputs include:

- `checkpoints/baseline_final.pth`
- `results/metrics/baseline.json`
- `results/figures/baseline_accuracy.png`
- `results/figures/baseline_loss.png`

### EXP02 - Transfer Learning

Frozen backbone:

```powershell
python -m scripts.train --strategy frozen --epochs 10 --batch-size 32
```

Partial fine-tuning:

```powershell
python -m scripts.train --strategy partial_finetune --epochs 3 --batch-size 32
```

Outputs include:

- `checkpoints/resnet18_frozen_best.pth`
- `checkpoints/resnet18_partial_finetune_best.pth`
- `results/metrics/frozen.json`
- `results/metrics/partial_finetune.json`

### EXP03 - Data Augmentation

```powershell
python -m scripts.train_augmentation
```

Outputs include:

- `checkpoints/augmentation_final.pth`
- `results/metrics/augmentation.json`
- `results/figures/augmentation_accuracy.png`
- `results/figures/augmentation_loss.png`

### EXP04 - Training-Data Size

```powershell
python -m scripts.train_data_size
```

Training subsets:

| Images per Class | Training Images |
|---:|---:|
| 5 | 1,000 |
| 10 | 2,000 |
| 15 | 3,000 |
| 20 | 4,000 |

Validation and official test sets stay fixed.

### Sampling Tests

```powershell
python -m experiments.exp04_data_size.test_dataset_sampling
```

The tests verify balanced sampling, reproducibility, error handling, and fixed validation/test sets.

## Final Model Evaluation

The final evaluation script computes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision / Recall / F1
- Confusion matrix

### BaselineCNN

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/baseline_final.pth --name baseline
```

### ResNet18 Frozen

```powershell
python "EVAL part/EVAL.py" --model-type resnet18_frozen --checkpoint checkpoints/resnet18_frozen_best.pth --name resnet18_frozen
```

### ResNet18 Partial Fine-Tuning

```powershell
python "EVAL part/EVAL.py" --model-type resnet18_partial --checkpoint checkpoints/resnet18_partial_finetune_best.pth --name resnet18_partial
```

### Augmentation

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/augmentation_final.pth --name augmentation
```

### Limited-Data Models

```powershell
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_5.pth --name data_size_5
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_10.pth --name data_size_10
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_15.pth --name data_size_15
python "EVAL part/EVAL.py" --model-type baseline --checkpoint checkpoints/data_size_20.pth --name data_size_20
```

Evaluation outputs are saved under:

```text
results/final_evaluation/
```

The verified experimental summary is also available in:

```text
EVAL part/evaluation_report.md
```

## Reproducibility Notes and Limitations

- Seed 42 is fixed for dataset splitting and sampling.
- Most experiment configurations were run once rather than across multiple random seeds, so mean +/- standard deviation is not reported.
- BaselineCNN and ResNet18 differ in both architecture and initialization, so their full accuracy gap is not a pure measurement of pretraining.
- Frozen and partial fine-tuning use different training hyperparameters and schedules.
- The selected augmentation policy is intentionally basic and produced only a small observed change.
- The model is a closed-set 200-class classifier and cannot reject unknown species or non-bird inputs.

## Final Project Report

The final report is stored under:

```text
report/Group6_Project2_Report.pdf
```

## Summary

The experiments support three main observations:

1. Increasing labeled training data generally improves the scratch baseline.
2. ImageNet-pretrained ResNet18 is much stronger than the simple scratch baseline in this limited-data setting.
3. Basic augmentation produced only a very small observed improvement in this setup.

The project is intended as an experimental study of limited-data FGIR rather than a state-of-the-art accuracy benchmark.
