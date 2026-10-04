
from pathlib import Path

import torch

from src.models.resnet18 import build_resnet18


def main():
    source_path = Path("checkpoints/resnet18_partial_finetune_best.pth")
    output_path = Path("checkpoints/resnet18_partial_full.pth")

    if not source_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source_path}")

    # Load the original fine-tuned checkpoint.
    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
    )

    # Restore the pretrained backbone.
    model = build_resnet18(
        num_classes=200,
        strategy="finetune",
    )

    # Restore the fine-tuned layers.
    model.layer4.load_state_dict(checkpoint["layer4_state_dict"])
    model.fc.load_state_dict(checkpoint["fc_state_dict"])

    # Save all layers, including the pretrained backbone.
    torch.save(model.state_dict(), output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"Exported model: {output_path}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
