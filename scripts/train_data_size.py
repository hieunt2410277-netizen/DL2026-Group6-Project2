import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import train_one_epoch, validate
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.results import save_metrics
from src.evaluation.visualization import plot_training_curves
from src.data.dataset import get_dataloaders


SAMPLE_LEVELS = [5, 10, 15, 20]


def run_experiment(samples_per_class, config, device):
    seed = config["seed"]
    set_seed(seed)

    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    print("\n" + "=" * 60)
    print(f"DATA SIZE EXPERIMENT: {samples_per_class} samples/class")
    print("=" * 60)

    train_loader, val_loader, test_loader, classes, num_classes = get_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        image_size=image_size,
        samples_per_class=samples_per_class,
        use_augmentation=False,
        seed=seed,
        num_workers=0,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = BaselineCNN(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "experiment": "data_size",
        "samples_per_class": samples_per_class,
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": None,
        "best_val_accuracy": None,
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

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"[{samples_per_class}/class] "
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

    history["best_val_accuracy"] = max(history["val_accuracy"])

    test_metrics = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    history["test_accuracy"] = test_metrics["accuracy"]

    print(
        f"[{samples_per_class}/class] "
        f"Best Val Acc: {history['best_val_accuracy']:.4f} "
        f"Test Acc: {history['test_accuracy']:.4f}"
    )

    torch.save(
        model.state_dict(),
        f"checkpoints/data_size_{samples_per_class}.pth"
    )

    save_metrics(
        history,
        f"results/metrics/data_size_{samples_per_class}.json"
    )

    plot_training_curves(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        train_accuracy=history["train_accuracy"],
        val_accuracy=history["val_accuracy"],
        output_path=f"results/figures/data_size_{samples_per_class}"
    )


def main():
    config = load_config("configs/baseline.yaml")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    for samples_per_class in SAMPLE_LEVELS:
        run_experiment(
            samples_per_class=samples_per_class,
            config=config,
            device=device,
        )


if __name__ == "__main__":
    main()