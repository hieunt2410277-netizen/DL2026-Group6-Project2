# Transfer Learning on CUB-200-2011

This project evaluates ImageNet-pretrained **ResNet18** on the CUB-200-2011
fine-grained bird classification task.

The implemented experiments are:

1. **Frozen backbone** — only the final classifier is trained.
2. **Partial fine-tuning** — `conv1` through `layer3` are frozen; `layer4`
   and the final classifier are trained.

> The project does not currently run full end-to-end fine-tuning of all ResNet18
> layers as a reported experiment.

## Dataset

Place the CUB-200-2011 dataset under:

```text
data/CUB_200_2011/
```

or place the original archive at:

```text
data/CUB_200_2011.tgz
```

See [`DATA.md`](DATA.md) for the complete dataset protocol.

## Install

From the project root:

```powershell
python -m pip install -r requirements.txt
```

## Frozen backbone

```powershell
python -m scripts.train --strategy frozen --epochs 10 --batch-size 32
```

Checkpoint:

```text
checkpoints/resnet18_frozen_best.pth
```

## Partial fine-tuning

```powershell
python -m scripts.train --strategy partial_finetune --epochs 3 --batch-size 32
```

Checkpoint:

```text
checkpoints/resnet18_partial_finetune_best.pth
```

The partial fine-tuning implementation caches the frozen `conv1`–`layer3`
features once. This reduces repeated CPU computation during later epochs.

## Reported results

| Strategy | Epochs | Trainable part | Best validation accuracy | Test accuracy |
|---|---:|---|---:|---:|
| Frozen backbone | 10 | Final `fc` | 50.92% | 50.88% |
| Partial fine-tuning | 3 | `layer4` + `fc` | 53.92% | **55.26%** |

These runs use different training schedules, so the comparison is descriptive
rather than a controlled equal-budget ablation.

## Compare results

```powershell
python -m scripts.compare_results
```

Regenerate figures:

```powershell
python -m scripts.plot_results
```

Outputs are stored in:

```text
results/metrics/
results/figures/
```

## Final evaluation

Use:

```powershell
python "EVAL part/EVAL.py" --model-type resnet18_frozen --checkpoint checkpoints/resnet18_frozen_best.pth --name resnet18_frozen

python "EVAL part/EVAL.py" --model-type resnet18_partial --checkpoint checkpoints/resnet18_partial_finetune_best.pth --name resnet18_partial
```

The evaluation pipeline reports accuracy, macro/weighted precision, recall,
F1, and a confusion matrix.

## CPU training

The implementation supports CPU execution. If a CUDA-capable NVIDIA GPU is
available, a compatible CUDA PyTorch installation can significantly reduce
training time.

The partial fine-tuning experiment includes CPU-specific feature caching to
avoid recomputing the frozen early ResNet layers on every epoch.
