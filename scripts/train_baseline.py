import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import train_one_epoch, validate
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.results import save_metrics
from src.evaluation.visualization import plot_training_curves


def main():
    config = load_config("configs/baseline.yaml")

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    print(f"Using device: {device}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {seed}")

    # TODO: Replace these when the common DataLoader is ready.
    train_loader = None
    val_loader = None
    num_classes = None

    if train_loader is None or val_loader is None:
        print(
            "Baseline training pipeline is waiting "
            "for the common DataLoader."
        )
        return

    model = BaselineCNN(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "experiment": "baseline",
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": None,
        "f1_score": None,
    }

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

    save_metrics(
        history,
        "results/metrics/baseline.json"
    )
    plot_training_curves(
    train_loss=history["train_loss"],
    val_loss=history["val_loss"],
    train_accuracy=history["train_accuracy"],
    val_accuracy=history["val_accuracy"],
    output_path="results/figures/baseline"
)


if __name__ == "__main__":
    main()