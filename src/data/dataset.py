from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.augmentation import get_transform


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
if __name__ == "__main__":
    import os
    
    # Tạo thư mục và file ảnh .jpg giả lập để ImageFolder không báo lỗi
    for split in ["train", "val", "test"]:
        os.makedirs(f"data/{split}/dummy_class", exist_ok=True)
        with open(f"data/{split}/dummy_class/dummy_image.jpg", "w") as f:
            f.write("")

    print("Đang khởi tạo DataLoader...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        batch_size=32
    )
    print(f"Thành công! Batch size tập train: {train_loader.batch_size}")
    print(f"Các lớp (classes) tìm thấy: {classes}")
    import os
    
    # Tạo thư mục giả lập để test DataLoader không bị báo lỗi FileNotFoundError
    for split in ["train", "val", "test"]:
        os.makedirs(f"data/{split}/dummy_class", exist_ok=True)

    print("Đang khởi tạo DataLoader...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        batch_size=32
    )
    print(f"Thành công! Batch size tập train: {train_loader.batch_size}")