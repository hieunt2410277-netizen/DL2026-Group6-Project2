import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(METRICS / "frozen.json", encoding="utf-8") as f:
    frozen = json.load(f)
with open(METRICS / "partial_finetune.json", encoding="utf-8") as f:
    partial = json.load(f)

def save_curve(values1, values2, labels, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    x1 = range(1, len(values1) + 1)
    x2 = range(1, len(values2) + 1)
    plt.plot(x1, values1, marker="o", label=labels[0])
    plt.plot(x2, values2, marker="o", label=labels[1])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200)
    plt.close()

save_curve(frozen["train_loss"], frozen["val_loss"],
           ["Train loss", "Validation loss"],
           "Frozen Backbone - Loss", "Loss", "frozen_loss.png")
save_curve(frozen["train_accuracy"], frozen["val_accuracy"],
           ["Train accuracy", "Validation accuracy"],
           "Frozen Backbone - Accuracy", "Accuracy", "frozen_accuracy.png")
save_curve(partial["train_loss"], partial["val_loss"],
           ["Train loss", "Validation loss"],
           "Partial Fine-Tuning - Loss", "Loss", "partial_finetune_loss.png")
save_curve(partial["train_accuracy"], partial["val_accuracy"],
           ["Train accuracy", "Validation accuracy"],
           "Partial Fine-Tuning - Accuracy", "Accuracy", "partial_finetune_accuracy.png")

plt.figure(figsize=(8, 5))
labels = ["Frozen", "Partial Fine-Tuning"]
vals = [frozen["best_val_accuracy"], partial["best_val_accuracy"]]
plt.bar(labels, vals)
plt.ylabel("Best validation accuracy")
plt.title("Transfer Learning Strategy Comparison")
plt.ylim(0, 1)
for i, v in enumerate(vals):
    plt.text(i, v + 0.02, f"{v:.2%}", ha="center")
plt.tight_layout()
plt.savefig(OUT / "strategy_comparison.png", dpi=200)
plt.close()

print(f"Saved figures to {OUT}")
