from collections import defaultdict
from pathlib import Path
import random
import tarfile

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import get_transform


DATASET_DIR_NAME = "CUB_200_2011"
ARCHIVE_NAME = "CUB_200_2011.tgz"

DEFAULT_SEED = 42


def prepare_dataset(data_dir: str = "./data") -> Path:
    """
    Ensure that CUB-200-2011 is extracted.

    The archive must be downloaded manually and placed at:
        data/CUB_200_2011.tgz
    """

    data_dir = Path(data_dir)

    archive_path = data_dir / ARCHIVE_NAME
    dataset_root = data_dir / DATASET_DIR_NAME
    images_dir = dataset_root / "images"

    if images_dir.exists():
        return dataset_root

    if not archive_path.exists():
        raise FileNotFoundError(
            f"CUB-200-2011 archive was not found at:\n"
            f"{archive_path}\n\n"
            f"Please download {ARCHIVE_NAME} from the official "
            f"CaltechDATA page and place it inside the data directory."
        )

    print(f"Extracting {archive_path}...")

    data_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        # Python 3.12+ supports extractall(filter="data"), but Python 3.11
        # does not. Validate archive members explicitly so the project works
        # on the documented Python versions without allowing path traversal.
        data_root = data_dir.resolve()

        for member in tar.getmembers():
            member_path = (data_dir / member.name).resolve()

            if os.path.commonpath([data_root, member_path]) != str(data_root):
                raise RuntimeError(
                    f"Unsafe archive member path detected: {member.name}"
                )

            if member.issym() or member.islnk():
                link_target = (member_path.parent / member.linkname).resolve()
                if os.path.commonpath([data_root, link_target]) != str(data_root):
                    raise RuntimeError(
                        f"Unsafe archive link detected: {member.name}"
                    )

        tar.extractall(path=data_dir)

    if not images_dir.exists():
        raise RuntimeError(
            "Dataset extraction completed, but the images directory "
            "could not be found."
        )

    print("Dataset extracted successfully.")

    return dataset_root


def _read_mapping(file_path: Path):
    mapping = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            key, value = line.strip().split(maxsplit=1)
            mapping[int(key)] = value

    return mapping


def load_metadata(dataset_root: Path):
    """
    Load official CUB-200-2011 metadata.
    """

    image_paths = _read_mapping(dataset_root / "images.txt")
    labels = _read_mapping(dataset_root / "image_class_labels.txt")
    official_split = _read_mapping(dataset_root / "train_test_split.txt")
    class_names = _read_mapping(dataset_root / "classes.txt")

    records = []

    for image_id, relative_path in image_paths.items():
        label = int(labels[image_id]) - 1

        is_train = int(official_split[image_id]) == 1

        records.append(
            {
                "image_id": image_id,
                "path": dataset_root / "images" / relative_path,
                "label": label,
                "is_train": is_train,
            }
        )

    classes = [
        class_names[class_id]
        for class_id in sorted(class_names)
    ]

    return records, classes


def split_official_train(
    records,
    val_split: float = 0.2,
    seed: int = DEFAULT_SEED,
):
    """
    Preserve the official test set.

    Only the official training set is divided into:
        train
        validation

    The split is performed per class to maintain class balance.
    """

    official_train = [
        record for record in records
        if record["is_train"]
    ]

    official_test = [
        record for record in records
        if not record["is_train"]
    ]

    records_by_class = defaultdict(list)

    for record in official_train:
        records_by_class[record["label"]].append(record)

    rng = random.Random(seed)

    train_records = []
    val_records = []

    for label, class_records in records_by_class.items():
        class_records = class_records.copy()

        rng.shuffle(class_records)

        val_count = max(
            1,
            int(round(len(class_records) * val_split))
        )

        val_records.extend(class_records[:val_count])
        train_records.extend(class_records[val_count:])

    return train_records, val_records, official_test


def limit_samples_per_class(
    records,
    samples_per_class: int | None,
    seed: int = DEFAULT_SEED,
):
    """
    Limit training samples independently for every class.

    Example:
        samples_per_class=10

    means at most 10 training images are used for each class.
    """

    if samples_per_class is None:
        return records

    if samples_per_class <= 0:
        raise ValueError(
            "samples_per_class must be greater than zero."
        )

    records_by_class = defaultdict(list)

    for record in records:
        records_by_class[record["label"]].append(record)

    insufficient_classes = [
        (label, len(class_records))
        for label, class_records in records_by_class.items()
        if len(class_records) < samples_per_class
    ]

    if insufficient_classes:
        label, available = min(
            insufficient_classes,
            key=lambda item: item[1],
        )
        raise ValueError(
            f"Cannot sample {samples_per_class} images per class: "
            f"class {label} has only {available} training images."
        )

    rng = random.Random(seed)

    limited_records = []

    for label, class_records in records_by_class.items():
        class_records = class_records.copy()

        rng.shuffle(class_records)

        limited_records.extend(
            class_records[:samples_per_class]
        )

    expected_count = samples_per_class * len(records_by_class)
    if len(limited_records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} sampled records, "
            f"got {len(limited_records)}."
        )

    return limited_records


class CUB200Dataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(record["path"]).convert("RGB")
        label = record["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    samples_per_class: int | None = None,
    use_augmentation: bool = False,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
):
    dataset_root = prepare_dataset(data_dir)

    records, classes = load_metadata(dataset_root)

    train_records, val_records, test_records = (
        split_official_train(
            records=records,
            val_split=val_split,
            seed=seed,
        )
    )

    train_records = limit_samples_per_class(
        records=train_records,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    augmentation_type = "basic" if use_augmentation else "none"

    train_transform = get_transform(
        augmentation=augmentation_type,
        image_size=image_size,
        train=True,
    )

    evaluation_transform = get_transform(
        augmentation="none",
        image_size=image_size,
        train=False,
    )

    train_dataset = CUB200Dataset(
        records=train_records,
        transform=train_transform,
    )

    val_dataset = CUB200Dataset(
        records=val_records,
        transform=evaluation_transform,
    )

    test_dataset = CUB200Dataset(
        records=test_records,
        transform=evaluation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    num_classes = len(classes)

    return (
        train_loader,
        val_loader,
        test_loader,
        classes,
        num_classes,
    )


if __name__ == "__main__":
    (
        train_loader,
        val_loader,
        test_loader,
        classes,
        num_classes,
    ) = get_dataloaders()

    images, labels = next(iter(train_loader))

    print(f"Number of classes: {num_classes}")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    print(
        f"Label range: "
        f"{labels.min().item()} - {labels.max().item()}"
    )