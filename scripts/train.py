import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import get_dataloaders
from src.models.resnet18 import build_resnet18, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def get_resnet_parts(model):
    """Return the frozen body through layer3, plus layer4/avgpool/fc."""
    children = list(model.children())
    # ResNet18 children:
    # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    frozen_body = nn.Sequential(*children[:7])
    layer4 = children[7]
    avgpool = children[8]
    fc = children[9]
    return frozen_body, layer4, avgpool, fc


@torch.no_grad()
def extract_frozen_features(backbone, loader, device):
    """Run the frozen part once and cache its output on CPU."""
    backbone.eval()
    features = []
    labels = []
    start = time.time()

    for images, y in loader:
        x = backbone(images.to(device))
        features.append(x.cpu())
        labels.append(y.cpu())

    return torch.cat(features), torch.cat(labels), time.time() - start


class PartialFineTuneModel(nn.Module):
    """Train only ResNet18 layer4 and the final classifier."""

    def __init__(self, layer4, avgpool, fc):
        super().__init__()
        self.layer4 = layer4
        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def save_history(history, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def train_frozen(args, device):
    """Original frozen-backbone experiment, kept unchanged in behavior."""
    lr = args.lr if args.lr is not None else 1e-3
    output_dir = Path(args.output_dir or "experiments/exp02_transfer_learning/frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        use_augmentation=True,
    )

    model = build_resnet18(num_classes, "frozen").to(device)
    total, trainable = count_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    history = {
        "strategy": "frozen",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "strategy": "frozen",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                },
                "checkpoints/resnet18_frozen_best.pth",
            )

    save_history(history, output_dir, "history.json")
    save_history(history, "results/metrics", "frozen.json")

    print("\nFrozen training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


def train_partial_finetune(args, device):
    """
    CPU-optimized partial fine-tuning:
      - frozen: conv1 through layer3
      - trainable: layer4 + final FC
      - cache frozen features once

    This avoids running/backpropagating through the first 3 ResNet stages
    during every training epoch.
    """
    lr = args.lr if args.lr is not None else 1e-4
    output_dir = Path(
        args.output_dir
        or "experiments/exp02_transfer_learning/partial_finetune"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    print("Strategy: partial fine-tuning")
    print("Frozen: conv1 through layer3")
    print("Trainable: layer4 + final FC")
    print("CPU optimization: cache frozen features once")

    # Deterministic transforms are used here because cached features are reused
    # across all epochs. Random augmentation would only happen once during caching.
    train_loader, val_loader, test_loader, classes, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        use_augmentation=False,
    )

    model = build_resnet18(num_classes, "finetune").to(device)
    frozen_body, layer4, avgpool, fc = get_resnet_parts(model)

    # Freeze everything through layer3.
    for param in frozen_body.parameters():
        param.requires_grad = False

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    cache_dir = Path("data/processed/cub200_partial_finetune")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train.pt"
    val_cache = cache_dir / "val.pt"

    if train_cache.exists() and val_cache.exists():
        print("Loading cached layer3 features...")
        train_x, train_y = torch.load(
            train_cache, map_location="cpu", weights_only=False
        )
        val_x, val_y = torch.load(
            val_cache, map_location="cpu", weights_only=False
        )
    else:
        print("Extracting train features (one-time step)...")
        train_x, train_y, elapsed = extract_frozen_features(
            frozen_body, train_loader, device
        )
        print(f"Train extraction: {elapsed:.1f}s")

        print("Extracting validation features (one-time step)...")
        val_x, val_y, elapsed = extract_frozen_features(
            frozen_body, val_loader, device
        )
        print(f"Validation extraction: {elapsed:.1f}s")

        torch.save((train_x, train_y), train_cache)
        torch.save((val_x, val_y), val_cache)
        print(f"Cached to {cache_dir}")

    train_features = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_features = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    partial_model = PartialFineTuneModel(layer4, avgpool, fc).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        partial_model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "strategy": "partial_finetune",
        "frozen": "conv1 through layer3",
        "trainable": "layer4 + fc",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "weight_decay": args.weight_decay,
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            partial_model,
            train_features,
            criterion,
            device,
            optimizer,
        )
        val_loss, val_acc = run_epoch(
            partial_model,
            val_features,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "strategy": "partial_finetune",
                    "frozen": "conv1 through layer3",
                    "trainable": "layer4 + fc",
                    "layer4_state_dict": layer4.state_dict(),
                    "fc_state_dict": fc.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                },
                "checkpoints/resnet18_partial_finetune_best.pth",
            )

    save_history(history, output_dir, "history.json")
    save_history(history, "results/metrics", "partial_finetune.json")

    print("\nPartial fine-tuning complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["frozen", "partial_finetune"],
        default="frozen",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--data-dir", default="data/raw/CUB_200_2011")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(args.cpu_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("CPU threads:", torch.get_num_threads())

    if args.strategy == "frozen":
        train_frozen(args, device)
    else:
        train_partial_finetune(args, device)


if __name__ == "__main__":
    main()
