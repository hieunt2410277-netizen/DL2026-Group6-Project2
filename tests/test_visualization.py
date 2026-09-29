from src.evaluation.visualization import plot_training_curves


def main():
    train_loss = [2.2, 1.8, 1.5, 1.2, 1.0]
    val_loss = [2.3, 1.9, 1.6, 1.5, 1.6]

    train_accuracy = [0.2, 0.4, 0.55, 0.7, 0.8]
    val_accuracy = [0.18, 0.35, 0.5, 0.58, 0.56]

    plot_training_curves(
        train_loss=train_loss,
        val_loss=val_loss,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        output_path="results/figures/baseline"
    )

    print("Training curves generated successfully.")


if __name__ == "__main__":
    main()
    