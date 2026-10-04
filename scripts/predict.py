
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import prepare_dataset, load_metadata
from src.data.augmentation import get_transform
from src.models.resnet18 import build_resnet18


def main():
    parser = argparse.ArgumentParser(
        description="Predict bird species using ResNet18 partial fine-tuning."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/resnet18_partial_finetune_best.pth",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.top_k < 1 or args.top_k > 200:
        raise ValueError("--top-k must be between 1 and 200")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Read the official class names in their original order.
    dataset_root = prepare_dataset("data")
    _, classes = load_metadata(dataset_root)
    num_classes = len(classes)

    # Build the ImageNet-pretrained backbone.
    model = build_resnet18(
        num_classes=num_classes,
        strategy="finetune",
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    # Restore the trained layers.
    model.layer4.load_state_dict(checkpoint["layer4_state_dict"])
    model.fc.load_state_dict(checkpoint["fc_state_dict"])
    model.eval()

    # Match the deterministic validation/test transformations.
    evaluation_transform = get_transform(
        augmentation="none",
        image_size=224,
        train=False,
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = evaluation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = probabilities.topk(args.top_k)

    print(f"\nImage: {image_path}")
    print(f"Device: {device}")
    print("\nTop predictions:")

    for rank, (score, index) in enumerate(zip(scores, indices), start=1):
        class_name = classes[index.item()]
        print(f"{rank}. {class_name}: {score.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
