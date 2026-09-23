import torch

from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    config = load_config("configs/baseline.yaml")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    seed = config["seed"]

    print(f"Using device: {device}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {seed}")

    print(
        "Baseline training pipeline is waiting "
        "for the common DataLoader."
    )


if __name__ == "__main__":
    main()