# Data Size Experiments Results

## Configuration

* Seed: 42
* Samples per class: 5, 10, 15, 20
* Validation split: fixed per-class split
* Test split: official test set kept fixed
* Learning rate: 0.001
* Epochs: 20

## Results

|Samples|Images|Accuracy|Best validation accuracy|F1 Score|
|-:|-:|-:|-:|-:|
|5|1000|1.97%|3.08%|1.18%|
|10|2000|2.93%|4.08%|2.10%|
|15|3000|3.38%|4.42%|2.37%|
|20|4000|4.90%|5.67%|3.67%|

## Observed trend

* Accuracy improved from 1.97% at 5 samples per class to 4.90% at 20 samples per class.
* Validation accuracy rose from 3.08% to 5.67%.
* F1 score increased from 1.18% to 3.67%.
* The improvement is gradual but consistent across all four settings.

## Conclusion

Reducing the available training data hurts classification performance. Increasing the number of labeled samples per class clearly improves both accuracy and F1 score, confirming that data volume is an important factor in fine-grained recognition.

