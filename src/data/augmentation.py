import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def no_augmentation_transform(image_size=224):
    """
    Transform for the baseline experiment without data augmentation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def basic_augmentation_transform(image_size=224):
    """
    Basic data augmentation:
    - Random resized crop
    - Random horizontal flip
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0)
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def strong_augmentation_transform(image_size=224):
    """
    Strong data augmentation:
    - Random resized crop
    - Random horizontal flip
    - Random rotation
    - Color jitter
    - Random erasing
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0)
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.RandomRotation(
            degrees=10
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        )
    ])


def validation_transform(image_size=224):
    """
    Deterministic transform for validation and test data.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_transform(
    augmentation="basic",
    image_size=224,
    train=True
):
    """
    Return the appropriate image transform.

    augmentation:
        "none", "basic", or "strong"

    train:
        True for training data.
        False for validation/test data.
    """

    if not train:
        return validation_transform(image_size)

    if augmentation == "none":
        return no_augmentation_transform(image_size)

    if augmentation == "basic":
        return basic_augmentation_transform(image_size)

    if augmentation == "strong":
        return strong_augmentation_transform(image_size)

    raise ValueError(
        f"Unknown augmentation type: {augmentation}. "
        "Choose from 'none', 'basic', or 'strong'."
    )