#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation metrics and functions.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.core.config import MODEL_DIR


def calculate_threshold_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold_method: str = 'f2',
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate metrics at different thresholds and find optimal threshold.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold_method: Method to use for finding optimal threshold ('f1', 'f2', 'custom')
        custom_threshold: Custom threshold to use if threshold_method is 'custom'
        
    Returns:
        Dictionary of metrics including best thresholds and associated metrics
    """
    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate precision-recall curve and PR-AUC
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate F1 and F2 scores for threshold optimization
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    f2_scores = 5 * precision * recall / (4 * precision + recall + 1e-10)
    
    # Find optimal F1 threshold
    best_f1_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    best_f1_threshold = thresholds[best_f1_idx] if len(thresholds) > 0 else 0.5
    
    # Find optimal F2 threshold (prioritizes recall more)
    best_f2_idx = np.argmax(f2_scores[:-1]) if len(thresholds) > 0 else 0
    best_f2_threshold = thresholds[best_f2_idx] if len(thresholds) > 0 else 0.5
    
    # Determine which threshold to use as primary
    if threshold_method == 'f1':
        best_threshold = best_f1_threshold
    elif threshold_method == 'f2':
        best_threshold = best_f2_threshold
    elif threshold_method == 'custom' and custom_threshold is not None:
        best_threshold = custom_threshold
    else:
        best_threshold = best_f2_threshold  # Default to F2
    
    # Calculate ROC curve and KS statistic
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks_score = max(tpr - fpr)
    
    # Test various thresholds
    threshold_results = []
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, best_f1_threshold, best_f2_threshold]
    
    # Add custom threshold if provided
    if custom_threshold is not None and custom_threshold not in test_thresholds:
        test_thresholds.append(custom_threshold)
    
    # Calculate metrics at each threshold
    for threshold in test_thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate F2 score
        f2 = 5 * prec * rec / (4 * prec + rec) if (prec + rec) > 0 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'f2': f2
        })
    
    # Apply best threshold and calculate final metrics
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    
    precision_at_best = precision_score(y_true, y_pred_best)
    recall_at_best = recall_score(y_true, y_pred_best)
    f1_at_best = f1_score(y_true, y_pred_best)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Compile all metrics
    metrics = {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'ks': ks_score,
        'best_threshold': best_threshold,
        'f1_threshold': best_f1_threshold,
        'f2_threshold': best_f2_threshold,
        'confusion_matrix': cm,
        'precision': precision_at_best,
        'recall': recall_at_best,
        'specificity': specificity,
        'f1': f1_at_best,
        'threshold_results': threshold_results,
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold_method: str = 'f2',
    custom_threshold: Optional[float] = None,
    model_name: str = 'model',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model predictions and generate evaluation report.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold_method: Method to use for threshold optimization
        custom_threshold: Custom threshold (optional)
        model_name: Name of the model for reporting
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate metrics
    metrics = calculate_threshold_metrics(
        y_true, 
        y_pred_proba, 
        threshold_method, 
        custom_threshold
    )
    
    # Output directory
    if output_dir is None:
        output_dir = MODEL_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_to_save = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    metrics_df = pd.DataFrame({
        'metric': list(metrics_to_save.keys()),
        'value': [str(v) for v in metrics_to_save.values()]
    })
    metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)
    
    # Generate a detailed report on thresholds
    threshold_df = pd.DataFrame(metrics['threshold_results'])
    threshold_df.to_csv(os.path.join(output_dir, f"{model_name}_thresholds.csv"), index=False)
    
    # Print summary metrics
    print(f"\n===== {model_name.upper()} EVALUATION SUMMARY =====")
    print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, KS: {metrics['ks']:.4f}")
    
    best_t = metrics['best_threshold']
    print(f"\nBest Threshold ({threshold_method}): {best_t:.4f}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}, F1 Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    return metrics


def compare_models(
    models_metrics: Dict[str, Dict[str, Any]],
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Args:
        models_metrics: Dictionary of model metrics {model_name: metrics_dict}
        output_dir: Output directory for saving results
        
    Returns:
        DataFrame with model comparison
    """
    # Define key metrics to compare
    key_metrics = ['auc', 'pr_auc', 'ks', 'precision', 'recall', 'f1', 'best_threshold']
    
    # Build comparison DataFrame
    comparison_data = []
    
    for model_name, metrics in models_metrics.items():
        model_results = {'model': model_name}
        
        for metric in key_metrics:
            if metric in metrics:
                model_results[metric] = metrics[metric]
        
        comparison_data.append(model_results)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by PR-AUC (or AUC if PR-AUC not available)
    if 'pr_auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('pr_auc', ascending=False)
    elif 'auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('auc', ascending=False)
    
    # Save comparison to file
    if output_dir is None:
        output_dir = MODEL_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Print comparison
    print("\n===== MODEL COMPARISON =====")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    return comparison_df


def calculate_lift(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate lift and cumulative lift.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calculating lift
        
    Returns:
        DataFrame with lift calculations
    """
    # Create a DataFrame with true labels and scores
    df = pd.DataFrame({
        'score': y_pred_proba,
        'target': y_true
    })
    
    # Sort by score in descending order
    df = df.sort_values('score', ascending=False)
    
    # Calculate overall positive rate
    overall_positive_rate = np.mean(y_true)
    
    # Create bins (equal number of samples in each bin)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False)
    
    # Calculate metrics for each bin
    bin_stats = []
    
    for bin_idx in range(n_bins):
        bin_df = df[df['bin'] == bin_idx]
        bin_size = len(bin_df)
        bin_positive = bin_df['target'].sum()
        bin_positive_rate = bin_positive / bin_size if bin_size > 0 else 0
        
        # Calculate lift
        lift = bin_positive_rate / overall_positive_rate if overall_positive_rate > 0 else 0
        
        bin_stats.append({
            'bin': bin_idx + 1,
            'size': bin_size,
            'positive_count': bin_positive,
            'positive_rate': bin_positive_rate,
            'lift': lift
        })
    
    # Calculate cumulative metrics
    lift_df = pd.DataFrame(bin_stats)
    lift_df['cumulative_size'] = lift_df['size'].cumsum()
    lift_df['cumulative_positive'] = lift_df['positive_count'].cumsum()
    lift_df['cumulative_positive_rate'] = lift_df['cumulative_positive'] / lift_df['cumulative_size']
    lift_df['cumulative_lift'] = lift_df['cumulative_positive_rate'] / overall_positive_rate
    
    # Calculate capture rate (% of all positives captured)
    total_positives = df['target'].sum()
    lift_df['capture_rate'] = lift_df['cumulative_positive'] / total_positives if total_positives > 0 else 0
    
    return lift_df


def plot_lift_chart(
    lift_df: pd.DataFrame,
    filepath: Optional[str] = None
) -> None:
    """
    Plot lift chart.
    
    Args:
        lift_df: DataFrame with lift calculations
        filepath: Output file path
    """
    import matplotlib.pyplot as plt
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot lift
    ax1.plot(lift_df['bin'], lift_df['lift'], 'b-', marker='o', label='Bin Lift')
    ax1.plot(lift_df['bin'], lift_df['cumulative_lift'], 'r-', marker='s', label='Cumulative Lift')
    ax1.axhline(y=1.0, color='k', linestyle='--')
    
    ax1.set_ylabel('Lift')
    ax1.set_title('Lift Chart')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add lift values as text annotations
    for i, (bin_num, lift_val, cum_lift) in enumerate(zip(lift_df['bin'], lift_df['lift'], lift_df['cumulative_lift'])):
        ax1.annotate(f'{lift_val:.2f}', xy=(bin_num, lift_val), xytext=(0, 5),
                    textcoords='offset points', ha='center')
        ax1.annotate(f'{cum_lift:.2f}', xy=(bin_num, cum_lift), xytext=(0, -15),
                    textcoords='offset points', ha='center')
    
    # Plot capture rate (cumulative % of positives captured)
    ax2.plot(lift_df['bin'], lift_df['capture_rate'], 'g-', marker='o')
    ax2.plot([1, len(lift_df)], [0, 1], 'k--')  # Diagonal line
    
    ax2.set_xlabel('Bin (Sorted by Predicted Probability)')
    ax2.set_ylabel('Capture Rate')
    ax2.set_title('Cumulative Capture Rate')
    ax2.grid(True, alpha=0.3)
    
    # Add capture rate values as text annotations
    for i, (bin_num, capture) in enumerate(zip(lift_df['bin'], lift_df['capture_rate'])):
        ax2.annotate(f'{capture:.2%}', xy=(bin_num, capture), xytext=(0, 5),
                    textcoords='offset points', ha='center')
    
    plt.tight_layout()
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'lift_chart.png')
        plt.savefig(output_path)
    
    plt.close() 