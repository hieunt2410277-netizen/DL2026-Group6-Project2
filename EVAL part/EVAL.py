import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# 1. Common Evaluation Metrics Definition
# ---------------------------------------------------------
def evaluate_model_predictions(y_true, y_pred, num_classes=200):
    """
    Computes Accuracy, Precision, Recall, and F1-score (Macro & Weighted).
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro: unweighted average across classes (suitable for balanced datasets)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted: weighted average based on sample count per class
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted)
    }
    return metrics

# ---------------------------------------------------------
# 2. Confusion Matrix & Training Curve Plots
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None, top_k_classes=20, save_path="confusion_matrix.png"):
    """
    Generates and saves a Confusion Matrix plot.
    For CUB-200 (200 classes), displays Top K most confused classes for clarity.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    if len(np.unique(y_true)) > top_k_classes:
        # Extract top_k_classes with highest misclassification counts
        errors_per_class = cm.sum(axis=1) - np.diag(cm)
        top_indices = np.argsort(errors_per_class)[-top_k_classes:]
        cm_sub = cm[np.ix_(top_indices, top_indices)]
        sub_names = [class_names[i] for i in top_indices] if class_names else top_indices
        
        sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sub_names, yticklabels=sub_names)
        plt.title(f'Confusion Matrix (Top {top_k_classes} Most Confused Classes)')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Full Confusion Matrix')
        
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plots Loss and Accuracy curves for Training and Validation across epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Curve
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    if train_acc and val_acc:
        ax2.plot(epochs, train_acc, 'b-o', label='Train Acc')
        ax2.plot(epochs, val_acc, 'r-s', label='Val Acc')
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------------------------------------------------------
# 3. Classification Error & Class Confusion Analysis
# ---------------------------------------------------------
def analyze_classification_errors(y_true, y_pred, class_names, top_n=10):
    """
    Extracts and ranks the most frequently confused class pairs.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0) # Exclude correct predictions
    
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))
                
    # Sort pairs by misclassification count descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n--- TOP {top_n} MOST CONFUSED CLASS PAIRS ---")
    for true_cls, pred_cls, count in pairs[:top_n]:
        print(f"True: {true_cls:<30} | Predicted as: {pred_cls:<30} | Count: {count}")
        
    return pairs[:top_n]