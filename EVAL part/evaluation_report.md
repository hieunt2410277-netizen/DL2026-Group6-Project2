# CUB-200-2011 Model Evaluation Report & Findings

## 1. Current Verified Results

| Experiment Group | Model / Setup | Best Validation Accuracy | Test Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Baseline | Custom BaselineCNN trained from scratch | 6.50% | Pending | Pending |
| Transfer Learning | ResNet18 frozen backbone | 58.28% | Pending | Pending |
| Transfer Learning | ResNet18 partial fine-tuning | 60.63% | Pending | Pending |
| Augmentation | Pending final experiment | Pending | Pending | Pending |
| Data Size | 5 / 10 / 15 / 20 samples per class | Pending | Pending | Pending |

## 2. Current Findings

### Baseline vs Transfer Learning

The custom CNN baseline achieved a best validation accuracy of 6.50%.

Using ImageNet-pretrained ResNet18 substantially improved validation accuracy:

- Frozen backbone: 58.28%
- Partial fine-tuning: 60.63%

These results indicate that transfer learning is much more effective than training the small baseline CNN from scratch on the limited CUB-200 training set.

### Augmentation

The augmentation pipeline has been integrated into the common data pipeline, but the final augmentation experiment has not yet been completed. No final accuracy or F1-score should be reported yet.

### Training Data Size

The limited-data sampler has been implemented and validated for controlled per-class sampling.

The intended experiment levels are:

- 5 samples/class
- 10 samples/class
- 15 samples/class
- 20 samples/class

Validation and official test sets remain fixed.

Final performance results are still pending.

## 3. Evaluation Still Required

The final evaluation stage still needs to:

- run each final model on the official CUB-200 test set;
- compute Accuracy;
- compute Macro Precision, Recall, and F1;
- generate confusion matrices;
- analyze the most confused class pairs;
- replace all pending entries in the comparison table.