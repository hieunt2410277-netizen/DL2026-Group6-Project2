# Transfer Learning on CUB-200-2011

This implementation covers:

- Pretrained ResNet18
- Replacement of the final classifier for 200 CUB classes
- Frozen-backbone training
- Full fine-tuning
- Common DataLoader
- Training/validation loss
- Training/validation accuracy
- Checkpoint saving
- Experiment configuration
- Result comparison

## 1. Dataset

Put the extracted CUB dataset here:

```text
data/raw/CUB_200_2011/
```

It should contain at least:

```text
images/
images.txt
image_class_labels.txt
train_test_split.txt
classes.txt
```

Do not commit the dataset to GitHub.

## 2. Install dependencies

```powershell
pip install torch torchvision pillow pyyaml
```

## 3. Test the DataLoader

From the project root:

```powershell
python scripts/test_data.py
```

Expected result is approximately:

```text
Images shape: torch.Size([32, 3, 224, 224])
Labels shape: torch.Size([32])
Training images: 5994
Validation/test images: 5794
```

## 4. Test the model

```powershell
python scripts/test_model.py
```

This checks both frozen and fine-tuning modes.

## 5. Run frozen-backbone training

```powershell
python scripts/train.py --strategy frozen --epochs 10 --batch-size 32
```

The best checkpoint is saved to:

```text
checkpoints/resnet18_frozen_best.pth
```

Metrics are saved to:

```text
results/metrics/frozen.json
```

## 6. Run fine-tuning

```powershell
python scripts/train.py --strategy finetune --epochs 10 --batch-size 32
```

The best checkpoint is saved to:

```text
checkpoints/resnet18_finetune_best.pth
```

## 7. Compare experiments

After both experiments finish:

```powershell
python scripts/compare_results.py
```

## Important: CPU training

If PyTorch reports:

```text
torch: ...+cpu
```

training will run on CPU and can be slow. If the computer has a compatible NVIDIA GPU, install a CUDA-enabled PyTorch build before running long experiments.

## Project mapping

The implementation corresponds to:

1. Load pretrained ResNet18
2. Replace final classification layer
3. Frozen-backbone training
4. Fine-tuning
5. Connect model to common DataLoader
6. Track training/validation loss
7. Track training/validation accuracy
8. Save experiment configurations
9. Run transfer-learning experiments
10. Compare frozen backbone and fine-tuning
11. Document settings and results
