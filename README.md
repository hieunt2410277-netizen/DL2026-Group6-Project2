# Fine-Grained Image Recognition under Limited Training Data

Deep Learning Course Project

## Overview

Fine-grained image recognition aims to distinguish between visually similar categories, such as different bird species, flower species, or dog breeds.

This project investigates how deep learning models perform when only a limited amount of labeled training data is available.

The project focuses on analyzing the effects of:

- Training data size
- Transfer learning
- Data augmentation
- Fine-tuning strategies

---


## Research Questions

This project investigates three main research questions:

1. **RQ1 - Training Data Size:** How does reducing the number of labeled training images affect fine-grained classification performance?
2. **RQ2 - Transfer Learning:** How effective is transfer learning when labeled training data is limited?
3. **RQ3 - Data Augmentation:** Does data augmentation improve generalization and reduce overfitting?

For RQ2, we additionally compare a frozen pretrained backbone with partial fine-tuning.


---


## Dataset

We use the **CUB-200-2011 (Caltech-UCSD Birds-200-2011)** dataset for fine-grained bird species classification.

- Number of classes: 200
- Total images: 11,788
- Official dataset: https://data.caltech.edu/records/65de6-vp158
- Official training images: 5,994
- Official test images: 5,794

The official training split is further divided into training and validation sets using a fixed random seed (42).

Our default data split is:

| Split | Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Test | 5,794 |

The official test set is kept separate from training and model development.


---


## Models

The project evaluates the following model configurations:

### 1. BaselineCNN

A custom convolutional neural network trained from scratch, without pretrained weights.

### 2. ResNet18 - Frozen Backbone

An ImageNet-pretrained ResNet18 with its feature extractor frozen. Only the final classification layer is trained.

### 3. ResNet18 - Partial Fine-Tuning

An ImageNet-pretrained ResNet18 where the earlier layers remain frozen while layer4 and the final classifier are trained.

### 4. BaselineCNN with Data Augmentation

The baseline architecture trained with basic image augmentation, including random cropping and horizontal flipping.


---


## Experiments

| Experiment | Description |
|---|---|
| EXP01 | Custom BaselineCNN trained from scratch |
| EXP02 | ResNet18 transfer learning: frozen vs partial fine-tuning |
| EXP03 | BaselineCNN with and without data augmentation |
| EXP04 | Impact of limited training data |

For EXP04, we use the following training-data sizes:

- 5 images per class (1,000 training images)
- 10 images per class (2,000 training images)
- 15 images per class (3,000 training images)
- 20 images per class (4,000 training images)

All data-size experiments use the same validation and official test sets.

### Main Results

| Model / Experiment | Test Accuracy |
|---|---:|
| BaselineCNN | 4.42% |
| ResNet18 Frozen | 50.88% |
| ResNet18 Partial Fine-Tuning | 55.26% |
| BaselineCNN with Augmentation | 4.57% |
| Data Size - 5 images/class | 1.85% |
| Data Size - 10 images/class | 3.07% |
| Data Size - 15 images/class | 3.38% |
| Data Size - 20 images/class | 4.90% |

The highest test accuracy is achieved by ResNet18 with partial fine-tuning.


---

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

---

## Project Structure

```text
Fine-Grained-Image-Recognition/
│
├── configs/
│   └── Experiment configuration files
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   └── Data exploration and visualization notebooks
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   └── Training and evaluation scripts
│
├── experiments/
│   ├── exp01_baseline/
│   ├── exp02_transfer_learning/
│   ├── exp03_augmentation/
│   └── exp04_data_size/
│
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── metrics/
│   └── README.md
│
├── checkpoints/
│   └── Local model weights
│
├── report/
│   └── Report materials
│
├── slides/
│   └── Presentation materials
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/hieunt2410277-netizen/Fine-Grained-Image-Recognition.git
cd Fine-Grained-Image-Recognition
```

### 2. Create a Virtual Environment

Python 3.11 or later is recommended.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The main dependencies include PyTorch, torchvision, NumPy, Pillow, matplotlib, scikit-learn and PyYAML.

### 4. Verify Installation

```powershell
python -c "import torch, torchvision, sklearn, PIL, yaml; print('Dependencies OK')"
```

## Dataset Setup

Download the CUB-200-2011 dataset from:

https://data.caltech.edu/records/65de6-vp158

Extract the dataset into:

`data/CUB_200_2011/`

Alternatively, place `CUB_200_2011.tgz` in the `data/` directory.

The data preparation pipeline will extract it automatically if needed.
For complete dataset documentation, including the official
data split, preprocessing, augmentation, and reproducibility
instructions, see [DATA.md](DATA.md).

### Verify Dataset Loading

Run from the repository root:

```powershell
python -m src.data.dataset
```

Expected dataset information:

```text
Number of classes: 200
Train samples: 4794
Validation samples: 1200
Test samples: 5794
Image batch shape: torch.Size([32, 3, 224, 224])
```

The project uses the official CUB-200-2011 test split and a fixed random seed of 42 for reproducibility.

For further dataset details, see [data/README.md](data/README.md).


---

## Reproducing Experiments

All commands below must be executed from the repository root after installing dependencies and preparing the CUB-200-2011 dataset.

The experiments were conducted with a fixed random seed of 42. The default image size is 224 × 224 pixels.

### EXP01 — Baseline CNN

Train the custom BaselineCNN from scratch:

```powershell
python -m scripts.train_baseline
```

Training configuration:

- Epochs: 20
- Batch size: 32
- Learning rate: 0.001
- Random seed: 42
- Data augmentation: disabled

Outputs:

- `checkpoints/baseline_final.pth`
- `results/metrics/baseline.json`
- `results/figures/baseline_accuracy.png`
- `results/figures/baseline_loss.png`

### EXP02 — Transfer Learning

The project evaluates two ImageNet-pretrained ResNet18 strategies.

**Frozen backbone (10 epochs):**

```powershell
python -m scripts.train --strategy frozen --epochs 10 --batch-size 32
```

**Partial fine-tuning (3 epochs):**

```powershell
python -m scripts.train --strategy partial_finetune --epochs 3 --batch-size 32
```

In the frozen strategy, only the final classification layer is trained.

In partial fine-tuning, the earlier layers are frozen while layer4 and the final classification layer are trained.

Outputs include:

- `checkpoints/resnet18_frozen_best.pth`
- `checkpoints/resnet18_partial_finetune_best.pth`
- `results/metrics/frozen.json`
- `results/metrics/partial_finetune.json`
- `experiments/exp02_transfer_learning/`

### EXP03 — Data Augmentation

Train the custom CNN with basic data augmentation:

```powershell
python -m scripts.train_augmentation
```

This experiment uses the same model architecture and main training settings as the baseline, but enables training-time augmentation.

Outputs:

- `checkpoints/augmentation_final.pth`
- `results/metrics/augmentation.json`
- `results/figures/augmentation_accuracy.png`
- `results/figures/augmentation_loss.png`

### EXP04 — Training Data Size

Train the custom CNN using different numbers of training images per class:

```powershell
python -m scripts.train_data_size
```

The script evaluates four data-size settings:

| Images per Class | Training Images |
|---:|---:|
| 5 | 1,000 |
| 10 | 2,000 |
| 15 | 3,000 |
| 20 | 4,000 |

The validation and official test sets remain unchanged.

Outputs:

- `checkpoints/data_size_5.pth`
- `checkpoints/data_size_10.pth`
- `checkpoints/data_size_15.pth`
- `checkpoints/data_size_20.pth`
- `results/metrics/data_size_*.json`
- `results/figures/data_size_*`

### Sampling Tests

Run the automated sampling tests:

```powershell
python -m experiments.exp04_data_size.test_dataset_sampling
```

The tests verify:

- Balanced per-class sampling
- Reproducible sampling with a fixed random seed
- Error handling for insufficient training images
- Fixed validation and test sets

Expected result: `Ran 4 tests` followed by `OK`.
## Inference Demo

The best-performing model in this project is ResNet18 with partial fine-tuning.

A complete model checkpoint is available through GitHub Releases. It contains all model weights, so downloading ImageNet pretrained weights is not required for inference.

**Download checkpoint:**

https://github.com/hieunt2410277-netizen/Fine-Grained-Image-Recognition/releases/download/v1.0.0-demo/resnet18_partial_full.pth

Place the downloaded file at:

`checkpoints/resnet18_partial_full.pth`

**Run inference:**

```powershell
python -m scripts.predict --image "path/to/bird.jpg" --checkpoint "checkpoints/resnet18_partial_full.pth" --top-k 5
```

The script outputs the Top-5 predicted bird species and their Softmax probabilities.

**Requirements:** Install dependencies from `requirements.txt`. The current inference script also requires the CUB-200-2011 dataset metadata to retrieve the official class names. Follow the Dataset Setup instructions before running inference.

**Model performance:** The partial fine-tuned ResNet18 achieved 55.26% accuracy on the official CUB-200-2011 test split.


## Final Model Evaluation

The final evaluation script runs inference on the official CUB-200-2011 test set.

It computes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted Precision, Recall and F1-score
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

`results/final_evaluation/`

Each evaluation produces a JSON file containing classification metrics and a confusion matrix image.

**Note:** Model checkpoints are stored locally and excluded from Git. To reproduce evaluation from a fresh clone, train the models first or obtain the corresponding checkpoints separately.

## Experimental Findings

The final verified results are summarized in:

`EVAL part/evaluation_report.md`

The strongest evaluated model is the ImageNet-pretrained ResNet18 with partial fine-tuning, achieving 55.26% test accuracy on CUB-200-2011.
