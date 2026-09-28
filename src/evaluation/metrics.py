import json
from pathlib import Path


def accuracy_from_logits(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
