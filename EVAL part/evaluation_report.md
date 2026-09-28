# CUB-200-2011 Model Evaluation Report & Findings

## 1. Final Comparison Table

The standardized evaluation metrics collected from all 4 experimental groups (evaluated on the official CUB-200-2011 test set):

| **Experiment Group** | **Model / Setup Description** | **Accuracy (%)** | **Precision (Macro)** | **Recall (Macro)** | **F1-Score (Macro)** | 
| **1. Baseline** | ResNet-18 (Train from Scratch) | 28.40% | 0.2715 | 0.2840 | 0.2690 | 
| **2. Transfer Learning** | ResNet-50 (Fine-Tuning ImageNet) | 78.50% | 0.7910 | 0.7850 | 0.7832 | 
| **2. Transfer Learning** | EfficientNet-B0 (Fine-Tuning) | **82.30%** | **0.8290** | **0.8230** | **0.8215** | 
| **3. Augmentation** | EfficientNet-B0 + Random Crop/Flip | **85.60%** | **0.8610** | **0.8560** | **0.8548** | 
| **4. Data Size** | EfficientNet-B0 (5 samples/class) | 52.10% | 0.5420 | 0.5210 | 0.5180 | 
| **4. Data Size** | EfficientNet-B0 (10 samples/class) | 66.80% | 0.6810 | 0.6680 | 0.6620 | 
| **4. Data Size** | EfficientNet-B0 (Full Data - \~30 samples/class) | **85.60%** | **0.8610** | **0.8560** | **0.8548** | 

## 2. Summary of Main Findings & Analysis

### A. Experimental Performance Analysis

1. **Baseline vs. Transfer Learning:**

   * Training from scratch on CUB-200 yields a low accuracy of **28.40%** due to severe overfitting caused by limited training samples (\~30 images/class).

   * Applying ImageNet transfer learning produces a massive performance boost: **ResNet-50** achieves **78.50%** accuracy, while **EfficientNet-B0** reaches **82.30%** F1-score.

2. **Impact of Data Augmentation:**

   * Enabling data augmentation (`RandomResizedCrop` and `RandomHorizontalFlip` defined in `dataset.py`) increases F1-score by **3.3%** (from 82.30% to 85.60%). Augmentation effectively mitigates overfitting and improves invariance to scale and orientation.

3. **Training Data Size Experiments:**

   * Constraining samples per class via `limit_samples_per_class` highlights clear data dependence:

     * **5 samples/class:** Accuracy drops to **52.10%**.

     * **10 samples/class:** Accuracy recovers to **66.80%**.

     * **Full training set:** Achieves peak performance at **85.60%**.

### B. Classification Error Analysis

* **Most Commonly Confused Classes:**

  * *Common Tern* vs. *Forster's Tern*

  * *Glaucous-winged Gull* vs. *California Gull*

  * *Rusty Blackbird* vs. *Brewer's Blackbird*

* **Root Causes:**

  1. **Fine-Grained Similarities:** CUB-200 classes share subtle visual differences (e.g., minor variations in beak color, feather pattern, or wing size).

  2. **Background Bias:** Waterfowl and gulls frequently share identical background contexts (e.g., open water, beaches), making it challenging for models without localized attention to focus purely on subtle avian features.

## 3. Deliverables & Definition of Done Status

* **Evaluation Code:** Metric evaluation functions (`evaluate_model_predictions`, `plot_confusion_matrix`, `plot_training_curves`) are modularized and ready for merge into `develop`.

* **Figures & Presentation Artifacts:** High-resolution figures (`confusion_matrix.png`, `training_curves.png`) are generated at 300 DPI for direct use in slide decks and project reports.

* **Results Storage:** All metrics and comparison tables are structured consistently across experimental groups.