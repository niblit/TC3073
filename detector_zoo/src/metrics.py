from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Any

def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluates model predictions against true labels and returns a dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def generate_report(metrics: Dict[str, float], seeds: Dict[str, Any]) -> str:
    """
    Generates a formatted baseline report string.
    """
    report = "========================================\n"
    report += "BASELINE MODEL EVALUATION REPORT\n"
    report += "========================================\n"
    report += "Reproducibility Parameters:\n"
    for k, v in seeds.items():
        report += f"  - {k}: {v}\n"
    report += "----------------------------------------\n"
    report += "Metrics:\n"
    report += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
    report += f"  Precision: {metrics['precision']:.4f}\n"
    report += f"  Recall:    {metrics['recall']:.4f}\n"
    report += f"  F1-Score:  {metrics['f1']:.4f}\n"
    report += "========================================\n"
    
    return report
