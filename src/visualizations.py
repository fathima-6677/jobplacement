import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_train_test_comparison(train_metrics, test_metrics, metric_names, output_dir):
    import numpy as np
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, train_metrics, width, label='Train')
    rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

    ax.set_ylabel('Scores')
    ax.set_title('Train vs Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_test_comparison.png'))
    plt.close()

def plot_class_distribution(y, output_dir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette='Set2')
    plt.title('Class Distribution')
    plt.xlabel('Status (0: Not Placed, 1: Placed)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
