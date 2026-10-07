# Transfer Learning Experiment Comparison

## Experimental results

| Strategy | Epochs | Trainable parameters | Best validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Frozen backbone | 10 | 102,600 | 50.92% | 50.88% |
| Partial fine-tuning (`layer4 + fc`) | 3 | 8,496,328 | 53.92% | **55.26%** |

Partial fine-tuning achieved the higher observed validation accuracy and the
best final test accuracy in the project.

The validation-accuracy difference is **3.00 percentage points**:

`53.92% - 50.92% = 3.00 percentage points`

The test-accuracy difference is **4.38 percentage points**:

`55.26% - 50.88% = 4.38 percentage points`

## Training curves

Generated figures are stored in `results/figures/`:

- `frozen_loss.png`
- `frozen_accuracy.png`
- `partial_finetune_loss.png`
- `partial_finetune_accuracy.png`
- `strategy_comparison.png`

## Interpretation

The frozen experiment trains only the final classification layer while keeping
the pretrained ResNet18 backbone fixed.

The partial fine-tuning experiment freezes `conv1` through `layer3` and updates
`layer4` together with the final classifier. This allows higher-level visual
features to adapt to CUB bird classes while retaining lower-level pretrained
features.

However, the runs use different epoch counts and learning rates (10 epochs at
`1e-3` versus 3 epochs at `1e-4`). Therefore, the results should be presented
as an observed comparison of the evaluated configurations, not as a controlled
causal measurement of the effect of fine-tuning alone.

## CPU optimization

The partial fine-tuning implementation caches the output of the frozen
`conv1`–`layer3` body once. Subsequent epochs train only `layer4` and `fc` on
the cached features, substantially reducing repeated CPU computation.

## Reproducibility

Both experiments use seed 42 for the shared dataset split. The reported runs
were performed once, so variation across multiple random seeds is not measured.
