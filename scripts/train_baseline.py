import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import train_one_epoch, validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    num_classes = 10
    batch_size = 8

    # Dummy dataset
    train_images = torch.randn(64, 3, 224, 224)
    train_labels = torch.randint(0, num_classes, (64,))

    val_images = torch.randn(32, 3, 224, 224)
    val_labels = torch.randint(0, num_classes, (32,))

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = BaselineCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_metrics = validate(
        model,
        val_loader,
        criterion,
        device
    )

    print()
    print("Train Results")
    print("Loss:", train_metrics["loss"])
    print("Accuracy:", train_metrics["accuracy"])

    print()
    print("Validation Results")
    print("Loss:", val_metrics["loss"])
    print("Accuracy:", val_metrics["accuracy"])


if __name__ == "__main__":
    main()