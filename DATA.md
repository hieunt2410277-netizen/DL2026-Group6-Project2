
# DATA.md — CUB-200-2011 Dataset Documentation

## 1. Dataset Information

**Dataset:** Caltech-UCSD Birds-200-2011 (CUB-200-2011)

**Dataset version:** CUB-200-2011, original 2011 release

**Task:** Fine-Grained Image Classification

**Number of classes:** 200 bird species

**Total images:** 11,788

Official dataset source:

https://data.caltech.edu/records/65de6-vp158

Original dataset website:

https://www.vision.caltech.edu/datasets/cub_200_2011/

Archive filename: `CUB_200_2011.tgz`

The project uses the original dataset and its official train/test split.

## 2. Download and Directory Structure

Download the official archive and extract it into the project's `data/` directory.

Expected structure:

```text
data/
└── CUB_200_2011/
    ├── images/
    ├── images.txt
    ├── image_class_labels.txt
    ├── train_test_split.txt
    └── classes.txt
```

Alternatively, place the original archive at:

`data/CUB_200_2011.tgz`

The data preparation code can extract the archive automatically.

Dataset image files are stored locally and excluded from Git.

## 3. Dataset Metadata

The project reads the official metadata files:

| File | Purpose |
|---|---|
| `images.txt` | Maps image IDs to image paths |
| `image_class_labels.txt` | Maps image IDs to class labels |
| `train_test_split.txt` | Defines the official train/test split |
| `classes.txt` | Contains the names of the 200 bird classes |

Original class labels range from 1 to 200.

For PyTorch classification, labels are converted to the range 0–199.

## 4. Data Splitting

The official CUB-200-2011 split contains:

- 5,994 official training images
- 5,794 official test images

The project divides the official training portion into training and validation subsets using a class-wise split.

**Validation fraction:** 20%

**Random seed:** 42

Final default dataset partitions:

| Split | Number of Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Test | 5,794 |
| Total | 11,788 |

The official test set remains unchanged.

Validation and test samples are not included in training.

## 5. Image Preprocessing

The default input image size is 224 × 224 pixels.

Preprocessing includes:

1. Loading each image using Pillow.
2. Converting images to RGB.
3. Resizing images to the required input dimensions for the non-augmented pipeline.
4. Converting images to PyTorch tensors.
5. Normalizing with ImageNet mean and standard deviation.

Normalization parameters:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Validation and test transformations are deterministic.

## 6. Data Augmentation

During the augmentation experiment, random transformations are applied to training images.

The basic augmentation strategy includes:

- Random resized cropping
- Random horizontal flipping

Augmentation is enabled only for the relevant training experiment.

Validation and test sets do not receive random training augmentation.

Implementation:

- `src/data/augmentation.py`
- `src/data/dataset.py`

## 7. Limited Training Data

To study the effect of limited labeled data, the project uses class-balanced sampling.

The supported experimental settings are:

| Samples per Class | Training Images |
|---:|---:|
| 5 | 1,000 |
| 10 | 2,000 |
| 15 | 3,000 |
| 20 | 4,000 |

Sampling is performed only on training records after the train/validation split.

The same seed is used for repeatable sampling.

Validation and test sets remain fixed across all data-size experiments.

## 8. Data Preparation and Reproduction Scripts

The main data preparation code is located at:

`src/data/dataset.py`

The common function is:

`get_dataloaders()`

To verify the default dataset split, run from the repository root:

```powershell
python -m src.data.dataset
```

Expected results:

```text
Number of classes: 200
Train samples: 4794
Validation samples: 1200
Test samples: 5794
Image batch shape: torch.Size([32, 3, 224, 224])
```

To test the reproducibility and correctness of limited-data sampling:

```powershell
python -m experiments.exp04_data_size.test_dataset_sampling
```

Expected result:

```text
Ran 4 tests
OK
```

Data-size experiments are executed with:

```powershell
python -m scripts.train_data_size
```

Training and evaluation commands for the other experiments are documented in the main `README.md`.

## 9. Dataset Reproducibility Notes

- The original CUB-200-2011 dataset is used.
- Official test images are preserved.
- Validation is created only from official training images.
- The default random seed is 42.
- Limited-data sampling operates per class.
- Validation and test sets remain unchanged when training-data size is reduced.
- No new dataset or redistributed processed image archive is produced by this project.
- Images and model checkpoints are excluded from Git due to file size.

For additional details, see `data/README.md`.
