# Transfer Learning Experiment Comparison

## Experimental results

| Strategy | Epochs | Trainable parameters | Best validation accuracy |
|---|---:|---:|---:|
| Frozen backbone | 10 | 102,600 | 58.28% |
| Partial fine-tuning (layer4 + FC) | 3 | 8,496,328 | 60.63% |

Partial fine-tuning achieved **60.63%** best validation accuracy, compared with **58.28%** for the frozen-backbone experiment. The observed absolute difference is **2.35 percentage points**.

## Training curves

The generated figures are stored in `results/figures/`:

- `frozen_loss.png`
- `frozen_accuracy.png`
- `partial_finetune_loss.png`
- `partial_finetune_accuracy.png`
- `strategy_comparison.png`

## Interpretation

The frozen experiment trains only the final classification layer while keeping the ResNet18 backbone fixed. The partial fine-tuning experiment updates the higher-level `layer4` features together with the final classifier, while keeping `conv1` through `layer3` frozen.

The partial fine-tuning run obtained the higher observed validation accuracy. However, the runs used different epoch counts (10 versus 3), so this should be reported as an observed result rather than a controlled equal-epoch comparison.

## CPU optimization

The partial fine-tuning implementation caches the output of the frozen part of ResNet18 once. Subsequent epochs operate on cached features and train only `layer4` and the final FC layer. In the measured run, each epoch took approximately 53.7–55.6 seconds on CPU.
