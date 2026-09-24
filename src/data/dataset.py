from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentation import get_transform


def create_dataloaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=32,
    image_size=224,
    augmentation="basic",
    num_workers=2
):
    """
    Create train, validation, and test DataLoaders.
    """

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)

    train_transform = get_transform(
        augmentation=augmentation,
        image_size=image_size,
        train=True
    )

    eval_transform = get_transform(
        augmentation=augmentation,
        image_size=image_size,
        train=False
    )

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )