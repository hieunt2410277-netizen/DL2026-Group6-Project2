import json
from pathlib import Path

files = [
    Path("results/metrics/frozen.json"),
    Path("results/metrics/partial_finetune.json"),
]

print("\nTransfer-learning comparison")
print("-" * 78)
print(f"{'Strategy':<25} {'Best Val Acc':<18} {'Final Val Loss':<18}")
print("-" * 78)

for path in files:
    if not path.exists():
        print(f"{path} not found - run that experiment first.")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    best_acc = max(data["val_accuracy"])
    final_loss = data["val_loss"][-1]

    print(f"{data['strategy']:<25} {best_acc:<18.4f} {final_loss:<18.4f}")
