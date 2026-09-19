from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import pandas as pd

def evaluate_predictions(y_true, y_pred, y_prob=None, dataset_name="Test"):
    print(f"--- {dataset_name} Evaluation ---")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        metrics['ROC-AUC'] = roc_auc
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics
