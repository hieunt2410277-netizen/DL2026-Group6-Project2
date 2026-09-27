from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CUB200Dataset(Dataset):
    """CUB-200-2011 Dataset using the official annotation files."""

    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        required = [
            "images.txt",
            "image_class_labels.txt",
            "train_test_split.txt",
            "images",
        ]
        missing = [x for x in required if not (self.root / x).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing CUB files in {self.root}: {', '.join(missing)}"
            )

        self.images = {}
        with open(self.root / "images.txt", "r") as f:
            for line in f:
                image_id, image_path = line.strip().split(maxsplit=1)
                self.images[int(image_id)] = image_path

        self.labels = {}
        with open(self.root / "image_class_labels.txt", "r") as f:
            for line in f:
                image_id, label = line.strip().split()
                self.labels[int(image_id)] = int(label) - 1

        self.samples = []
        with open(self.root / "train_test_split.txt", "r") as f:
            for line in f:
                image_id, is_train = line.strip().split()
                image_id = int(image_id)
                is_train = int(is_train)
                if (train and is_train == 1) or ((not train) and is_train == 0):
                    self.samples.append(image_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id = self.samples[index]
        image_path = self.root / "images" / self.images[image_id]

        image = Image.open(image_path).convert("RGB")
        label = self.labels[image_id]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_transforms(augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir, batch_size=32, num_workers=0, augment=True):
    train_transform, eval_transform = build_transforms(augment=augment)

    train_dataset = CUB200Dataset(
        root=data_dir, train=True, transform=train_transform
    )
    val_dataset = CUB200Dataset(
        root=data_dir, train=False, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
