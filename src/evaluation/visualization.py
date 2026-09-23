from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    train_loss,
    val_loss,
    train_accuracy,
    val_accuracy,
    output_path: str,
):
    epochs = range(1, len(train_loss) + 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure()

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    loss_path = output.parent / f"{output.stem}_loss.png"

    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()

    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    accuracy_path = output.parent / f"{output.stem}_accuracy.png"

    plt.savefig(accuracy_path, bbox_inches="tight")
    plt.close()