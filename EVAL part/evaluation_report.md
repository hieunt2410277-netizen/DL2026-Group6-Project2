# CUB-200-2011 Model Evaluation Report & Findings

## 1. Final Verified Experimental Results

All reported test metrics below were computed on the official CUB-200-2011 test set using the final trained checkpoints.

| Experiment Group | Model / Setup | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| Baseline | Custom BaselineCNN trained from scratch | 4.42% | 3.80% | 4.52% | 3.17% |
| Transfer Learning | ResNet18 frozen backbone | 50.88% | 55.40% | 51.23% | 51.40% |
| Transfer Learning | ResNet18 partial fine-tuning | 55.26% | 58.84% | 55.49% | 54.47% |
| Augmentation | BaselineCNN + basic augmentation | 4.57% | 5.02% | 4.68% | 3.36% |
| Data Size | BaselineCNN - 5 samples/class | 1.85% | 1.72% | 1.94% | 1.18% |
| Data Size | BaselineCNN - 10 samples/class | 3.07% | 2.81% | 3.12% | 2.10% |
| Data Size | BaselineCNN - 15 samples/class | 3.38% | 2.55% | 3.44% | 2.37% |
| Data Size | BaselineCNN - 20 samples/class | 4.90% | 4.16% | 4.95% | 3.67% |

## 2. Baseline Performance

The baseline experiment uses a small custom convolutional neural network trained from scratch on the CUB-200-2011 dataset.

Final test results:

- Accuracy: 4.42%
- Macro Precision: 3.80%
- Macro Recall: 4.52%
- Macro F1-score: 3.17%

The low baseline performance reflects the difficulty of fine-grained classification on 200 visually similar bird species when the model is trained from scratch with relatively limited labeled data.

## 3. Transfer Learning Analysis

Two ImageNet-pretrained ResNet18 transfer-learning strategies were evaluated.

### Frozen Backbone

Only the final classifier was trained while the pretrained feature extractor remained frozen.

Final test results:

- Accuracy: 50.88%
- Macro Precision: 55.40%
- Macro Recall: 51.23%
- Macro F1-score: 51.40%

### Partial Fine-Tuning

The early ResNet18 layers were frozen while layer4 and the final fully connected classifier were trained.

Final test results:

- Accuracy: 55.26%
- Macro Precision: 58.84%
- Macro Recall: 55.49%
- Macro F1-score: 54.47%

Partial fine-tuning achieved the highest performance among all evaluated models.

Compared with the baseline test accuracy of 4.42%, partial fine-tuning increased test accuracy to 55.26%.

These results demonstrate the effectiveness of pretrained ImageNet representations for fine-grained image recognition when labeled training data is limited.

## 4. Data Augmentation Analysis

The augmentation experiment used the same BaselineCNN architecture and training configuration as the baseline, while enabling basic image augmentation during training.

The augmentation pipeline included random crop and horizontal flip operations.

Final test results:

- Accuracy: 4.57%
- Macro Precision: 5.02%
- Macro Recall: 4.68%
- Macro F1-score: 3.36%

The baseline test accuracy was 4.42%, while the augmentation experiment achieved 4.57%.

This difference is small, therefore the current experiment does not provide strong evidence that the selected basic augmentation strategy substantially improves generalization for the BaselineCNN.

This result does not imply that augmentation is ineffective in general. Stronger augmentation strategies, different model architectures, or additional hyperparameter tuning may produce different results.

## 5. Training Data Size Analysis

The limited-data experiments controlled the number of training images available per class while keeping the validation and official test sets fixed.

The evaluated training-data sizes were:

- 5 images per class
- 10 images per class
- 15 images per class
- 20 images per class

### Results

| Samples per Class | Training Images | Test Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 5 | 1000 | 1.85% | 1.18% |
| 10 | 2000 | 3.07% | 2.10% |
| 15 | 3000 | 3.38% | 2.37% |
| 20 | 4000 | 4.90% | 3.67% |

Performance generally improved as more labeled images were provided.

Test accuracy increased from 1.85% at 5 samples per class to 4.90% at 20 samples per class.

The results show that fine-grained classification performance is strongly dependent on the amount of labeled training data available.

## 6. Answers to Research Questions

### RQ1. How does reducing training-data size affect classification performance?

Reducing the amount of labeled training data decreases classification performance.

The test accuracy increased consistently as the number of samples per class increased:

- 5 samples/class: 1.85%
- 10 samples/class: 3.07%
- 15 samples/class: 3.38%
- 20 samples/class: 4.90%

Macro F1-score also increased from 1.18% to 3.67%.

Therefore, the experiments show that increasing the amount of labeled training data improves generalization on CUB-200-2011.

### RQ2. How much does transfer learning help when labeled data is limited?

Transfer learning provides a substantial improvement over training the custom CNN from scratch.

The baseline achieved only 4.42% test accuracy.

Using pretrained ResNet18 increased performance to:

- 50.88% test accuracy with a frozen backbone
- 55.26% test accuracy with partial fine-tuning

Partial fine-tuning also achieved a Macro F1-score of 54.47%, compared with 3.17% for the baseline.

These results show that transfer learning is highly effective for fine-grained image recognition under limited labeled data.

### RQ3. Does data augmentation improve generalization and reduce overfitting?

The basic augmentation strategy produced only a small improvement over the baseline.

- Baseline test accuracy: 4.42%
- Augmentation test accuracy: 4.57%

Macro F1-score increased slightly from 3.17% to 3.36%.

Therefore, under the current BaselineCNN architecture and augmentation settings, the experiment does not show a clear or substantial improvement in generalization.

Further experiments with stronger augmentation strategies or pretrained models may produce different results.

## 7. Classification Error Analysis

Fine-grained recognition remains challenging because many CUB-200 classes have very similar visual characteristics.

The generated confusion matrices can be used to inspect the most frequently confused classes.

Potential sources of classification error include:

1. Similar bird species with only subtle differences in plumage, beak shape, body proportions, or feather patterns.
2. Variation in pose, scale, and viewpoint.
3. Background information that may distract the classifier from bird-specific features.
4. Limited training examples for each class.
5. The limited representation capacity of the small BaselineCNN compared with pretrained ResNet18.

The final evaluation pipeline generates confusion matrices for each evaluated checkpoint under:

`results/final_evaluation/`

## 8. Overall Conclusion

The experiments provide three main conclusions.

First, increasing the amount of labeled training data improves classification performance.

Second, transfer learning provides by far the largest performance improvement in this project. The best model, ResNet18 with partial fine-tuning, achieved 55.26% test accuracy compared with 4.42% for the custom baseline CNN.

Third, the selected basic augmentation strategy produced only a small improvement and did not substantially change generalization performance.

Overall, pretrained deep convolutional features are especially valuable for fine-grained image recognition when the available labeled dataset is limited.