# Dataset: CUB-200-2011

## 1. Dataset Overview

This project uses the **CUB-200-2011 (Caltech-UCSD Birds-200-2011)** dataset for fine-grained bird species classification.

The dataset contains visually similar bird species, making it suitable for evaluating fine-grained recognition under limited labeled training data.

### Dataset Statistics

| Property | Value |
|---|---|
| Dataset | CUB-200-2011 |
| Classes | 200 bird species |
| Total images | 11,788 |
| Official training images | 5,994 |
| Official test images | 5,794 |
| Random seed | 42 |

## 2. Dataset Source

Official dataset information:

https://data.caltech.edu/records/65de6-vp158

Original dataset website:

https://www.vision.caltech.edu/datasets/cub_200_2011/

Download the dataset archive named `CUB_200_2011.tgz`.

The original dataset is used without modifying its class labels or official test split.

## 3. Dataset Installation

Download and extract the archive into the project's `data/` directory.

The expected dataset location is:

`data/CUB_200_2011/`

The directory should contain:

- `images/`
- `images.txt`
- `image_class_labels.txt`
- `train_test_split.txt`
- `classes.txt`

Alternatively, place `CUB_200_2011.tgz` inside `data/`. The common data preparation pipeline can extract the archive when required.

**Note:** Dataset images and archives are not committed to GitHub because of their size.

## 4. Dataset Splitting

The project preserves the official CUB-200-2011 test split.

Only the official training portion is divided into training and validation subsets using class-wise splitting and random seed 42.

| Split | Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Test | 5,794 |
| Total | 11,788 |

The validation and test sets remain fixed during the limited-data experiments.

## 5. Data Preprocessing

Images are:

1. Converted to RGB.
2. Resized and cropped to the configured input size of 224 × 224 pixels.
3. Converted to PyTorch tensors.
4. Normalized using ImageNet mean and standard deviation.

Basic data augmentation can be enabled during training and includes random cropping and horizontal flipping.

Validation and test preprocessing does not use random augmentation.

## 6. Limited Training Data

For experiments investigating training-data size, the project samples a fixed number of training images independently from each class.

The evaluated settings are:

| Images per Class | Training Images |
|---|---:|
| 5 | 1,000 |
| 10 | 2,000 |
| 15 | 3,000 |
| 20 | 4,000 |

A fixed random seed is used to ensure reproducibility.

Only training images are sampled. Validation and test images are not modified.

## 7. Implementation

Dataset preparation and loading are implemented in:

- `src/data/dataset.py`
- `src/data/augmentation.py`

The common entry point is `get_dataloaders()`.

From the repository root, the data pipeline can be checked using:

`python -m src.data.dataset`

The expected output contains 200 classes, 4,794 training samples, 1,200 validation samples and 5,794 test samples.

## 8. Reproducibility

To reproduce the dataset configuration:

1. Download the official CUB-200-2011 dataset.
2. Extract it into `data/CUB_200_2011/`.
3. Install the project dependencies.
4. Run the common data pipeline using seed 42.
5. Keep the official test set unchanged for evaluation.

The same dataset partitions are used across the project's controlled experiments.