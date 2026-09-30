import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataset import get_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet18 import build_resnet18


def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def collect_predictions(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
    top_k=20,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    top_indices = np.argsort(errors_per_class)[-top_k:]

    cm_sub = cm[np.ix_(top_indices, top_indices)]
    names = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(cm_sub)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Top {top_k} most confused CUB-200 classes"
    )

    fig.colorbar(image, ax=ax)

    for i in range(cm_sub.shape[0]):
        for j in range(cm_sub.shape[1]):
            if cm_sub[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(cm_sub[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_model(model_type, checkpoint_path, num_classes, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if model_type == "baseline":
        model = BaselineCNN(
            num_classes=num_classes
        ).to(device)

        state_dict = checkpoint

    elif model_type == "resnet18_frozen":
        model = build_resnet18(
            num_classes=num_classes,
            strategy="frozen",
        ).to(device)

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    elif model_type == "resnet18_partial":
        model = build_resnet18(
            num_classes=num_classes,
            strategy="finetune",
        ).to(device)

        model.layer4.load_state_dict(
            checkpoint["layer4_state_dict"]
        )

        model.fc.load_state_dict(
            checkpoint["fc_state_dict"]
        )

        state_dict = None

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--model-type",
    choices=[
        "baseline",
        "resnet18_frozen",
        "resnet18_partial",
    ],
    required=True,
)

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--name",
        required=True,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Experiment: {args.name}")
    print(f"Checkpoint: {args.checkpoint}")

    _, _, test_loader, classes, num_classes = (
        get_dataloaders(
            data_dir="data",
            batch_size=32,
            image_size=224,
            samples_per_class=None,
            use_augmentation=False,
            seed=42,
            num_workers=0,
        )
    )

    model = load_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        device=device,
    )

    y_true, y_pred = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )

    output_dir = Path("results/final_evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = output_dir / f"{args.name}.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    cm_path = (
        output_dir
        / f"{args.name}_confusion_matrix.png"
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=classes,
        save_path=cm_path,
    )

    print("\nFinal test metrics")
    print("------------------")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()