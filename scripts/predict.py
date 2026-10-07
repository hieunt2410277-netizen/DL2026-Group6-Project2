
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

from src.data.augmentation import get_transform
from src.models.resnet18 import build_resnet18


def main():
    parser = argparse.ArgumentParser(
        description="Predict bird species using ResNet18 partial fine-tuning."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/resnet18_partial_full.pth",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Validate input files.
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load class names without requiring the dataset.
    classes_path = Path("configs/classes.txt")
    classes = {}

    with open(classes_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            class_id, class_name = line.strip().split(maxsplit=1)
            classes[int(class_id) - 1] = class_name

    num_classes = len(classes)

    if num_classes != 200 or set(classes) != set(range(200)):
        raise ValueError("Expected exactly 200 classes with IDs 1-200")

    if not 1 <= args.top_k <= num_classes:
        raise ValueError(
            f"--top-k must be between 1 and {num_classes}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Support both original partial and exported full checkpoints.
    if (
        "layer4_state_dict" in checkpoint
        and "fc_state_dict" in checkpoint
    ):
        print("Loading original partial checkpoint...")

        # Requires cached or downloadable ImageNet pretrained weights.
        model = build_resnet18(
            num_classes=num_classes,
            strategy="finetune",
        )
        model.layer4.load_state_dict(
            checkpoint["layer4_state_dict"]
        )
        model.fc.load_state_dict(
            checkpoint["fc_state_dict"]
        )
    else:
        print("Loading full checkpoint...")

        # Full checkpoint contains all model weights.
        # No ImageNet pretrained download is needed.
        model = resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Use the same preprocessing as validation/test.
    evaluation_transform = get_transform(
        augmentation="none",
        image_size=224,
        train=False,
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = evaluation_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = probabilities.topk(args.top_k)

    print(f"\nImage: {image_path}")
    print(f"Device: {device}")
    print("\nTop predictions:")

    for rank, (score, index) in enumerate(
        zip(scores, indices),
        start=1,
    ):
        class_name = classes[index.item()]
        probability = score.item() * 100
        print(f"{rank}. {class_name}: {probability:.2f}%")


if __name__ == "__main__":
    main()
