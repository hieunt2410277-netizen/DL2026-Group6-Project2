import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import train_one_epoch, validate


def test_baseline_forward_shape():
    model = BaselineCNN(num_classes=10)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 10)


def test_training_and_validation_pipeline():
    torch.manual_seed(42)
    device = torch.device("cpu")
    train_images = torch.randn(16, 3, 224, 224)
    train_labels = torch.randint(0, 10, (16,))
    val_images = torch.randn(8, 3, 224, 224)
    val_labels = torch.randint(0, 10, (8,))

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_labels),
        batch_size=4,
        shuffle=False,
    )

    model = BaselineCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_metrics = validate(
        model, val_loader, criterion, device
    )

    assert train_metrics["loss"] > 0
    assert 0 <= train_metrics["accuracy"] <= 1
    assert val_metrics["loss"] > 0
    assert 0 <= val_metrics["accuracy"] <= 1


if __name__ == "__main__":
    test_baseline_forward_shape()
    test_training_and_validation_pipeline()
    print("Baseline pipeline tests passed.")
