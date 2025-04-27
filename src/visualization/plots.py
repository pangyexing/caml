#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting functions for model visualization.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.core.config import MODEL_DIR
from src.utils import configure_fonts_for_plots

# Call the font configuration function at module load time
configure_fonts_for_plots()


def set_plotting_style():
    """Set consistent plotting style."""
    # Set seaborn style
    sns.set_style('whitegrid')
    
    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Use high-resolution figures
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


def plot_feature_importance(
    importance_dict: Dict[str, float], 
    title: str = 'Feature Importance', 
    top_n: int = 20,
    importance_type: str = 'gain',
    filepath: Optional[str] = None
) -> None:
    """
    Plot feature importance from the model.
    
    Args:
        importance_dict: Dictionary of feature importance values
        title: Plot title
        top_n: Number of top features to display
        importance_type: Type of importance (gain, weight, cover, etc.)
        filepath: Output file path
    """
    # Sort importance values
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Get top features
    plot_features = [item[0] for item in sorted_importance[:top_n]]
    plot_scores = [item[1] for item in sorted_importance[:top_n]]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(plot_scores)), plot_scores[::-1], align='center')
    plt.yticks(range(len(plot_scores)), plot_features[::-1])
    plt.xlabel(f'Importance ({importance_type})')
    plt.ylabel('Features')
    plt.title(f'{title} (Top {top_n})')
    plt.tight_layout()
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'feature_importance.png')
        plt.savefig(output_path)
    
    plt.close()


def plot_shap_summary(
    shap_values, 
    features, 
    title: str = 'SHAP Feature Impact', 
    max_display: int = 20,
    plot_type: str = 'dot',
    filepath: Optional[str] = None
) -> None:
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values from explainer
        features: Feature matrix (X)
        title: Plot title
        max_display: Maximum features to display
        plot_type: Plot type ('dot' or 'bar')
        filepath: Output file path
    """
    plt.figure(figsize=(14, 10))
    
    if plot_type == 'bar':
        shap.summary_plot(shap_values, features, plot_type='bar', show=False, max_display=max_display)
    else:
        shap.summary_plot(shap_values, features, show=False, max_display=max_display)
    
    plt.title(title)
    plt.tight_layout()
    
    # Save if filepath provided
    if filepath:
        # Make sure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300)
    else:
        output_path = os.path.join(MODEL_DIR, f'shap_summary_{plot_type}.png')
        os.makedirs(MODEL_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    plt.close()


def plot_precision_recall_curve(
    precision: np.ndarray, 
    recall: np.ndarray, 
    thresholds: np.ndarray,
    best_threshold: float,
    f1_threshold: float,
    pr_auc: float,
    filepath: Optional[str] = None
) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        thresholds: Threshold values
        best_threshold: Best threshold value (optimized for F2)
        f1_threshold: F1-optimized threshold
        pr_auc: Precision-recall AUC
        filepath: Output file path
    """
    plt.figure(figsize=(10, 8))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, 'b-', linewidth=2)
    
    # Mark thresholds
    threshold_idxs = []
    
    # Find indices for specific thresholds
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        idx = (np.abs(thresholds - t)).argmin()
        if idx < len(precision):
            threshold_idxs.append((idx, t))
    
    # Mark best thresholds
    best_idx = (np.abs(thresholds - best_threshold)).argmin()
    f1_idx = (np.abs(thresholds - f1_threshold)).argmin()
    
    threshold_idxs.extend([(best_idx, best_threshold), (f1_idx, f1_threshold)])
    
    # Plot markers for thresholds
    for idx, t in threshold_idxs:
        if idx < len(precision):
            plt.plot(recall[idx], precision[idx], 'ro')
            plt.annotate(f'{t:.2f}', 
                        xy=(recall[idx], precision[idx]),
                        xytext=(recall[idx]+0.02, precision[idx]-0.05),
                        arrowprops=dict(arrowstyle='->'))
    
    # Add PR-AUC and best threshold
    plt.title(f'Precision-Recall Curve (PR-AUC = {pr_auc:.4f})')
    plt.text(0.5, 0.3, f'Best Threshold (F2): {best_threshold:.4f}\nF1 Threshold: {f1_threshold:.4f}',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'precision_recall_curve.png')
        plt.savefig(output_path)
    
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray, 
    tpr: np.ndarray, 
    auc_score: float,
    filepath: Optional[str] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate values
        tpr: True positive rate values
        auc_score: AUC score
        filepath: Output file path
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    
    # Calculate KS statistic
    ks_score = max(tpr - fpr)
    ks_idx = np.argmax(tpr - fpr)
    
    # Mark KS point
    plt.plot([fpr[ks_idx], fpr[ks_idx]], [fpr[ks_idx], tpr[ks_idx]], 'r--')
    plt.plot(fpr[ks_idx], tpr[ks_idx], 'ro')
    
    # Add AUC and KS information
    plt.title(f'ROC Curve (AUC = {auc_score:.4f})')
    plt.text(0.6, 0.2, f'KS: {ks_score:.4f}',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'roc_curve.png')
        plt.savefig(output_path)
    
    plt.close()


def create_score_bins(
    scores: np.ndarray, 
    n_bins: int = 10, 
    labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create score bins for model scores.
    
    Args:
        scores: Model prediction scores
        n_bins: Number of bins
        labels: Bin labels (optional)
        
    Returns:
        bins: Bin edges
        bin_indices: Bin indices for each score
        bin_labels: Labels for each bin
    """
    # Create bin edges
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Assign each score to a bin
    bin_indices = np.digitize(scores, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
    
    # Create bin labels if not provided
    if labels is None:
        bin_labels = []
        for i in range(n_bins):
            bin_labels.append(f'{bins[i]:.2f}-{bins[i+1]:.2f}')
    else:
        bin_labels = labels
        
    return bins, bin_indices, bin_labels


def plot_score_distribution(
    scores: np.ndarray, 
    labels: np.ndarray,
    n_bins: int = 10, 
    title: str = 'Score Distribution', 
    filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot score distribution with positive rate by bin.
    
    Args:
        scores: Model prediction scores
        labels: True labels
        n_bins: Number of bins
        title: Plot title
        filepath: Output file path
        
    Returns:
        DataFrame with bin statistics
    """
    # Create bins and get bin indices
    bins, bin_indices, bin_labels = create_score_bins(scores, n_bins)
    
    # Calculate counts and positive rates for each bin
    bin_counts = np.zeros(n_bins)
    bin_pos_counts = np.zeros(n_bins)
    
    for i in range(len(scores)):
        bin_idx = bin_indices[i]
        bin_counts[bin_idx] += 1
        bin_pos_counts[bin_idx] += labels[i]
    
    # Calculate positive rate
    bin_pos_rate = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_pos_rate[i] = bin_pos_counts[i] / bin_counts[i]
    
    # Create DataFrame for results
    bin_df = pd.DataFrame({
        'bin_label': bin_labels,
        'min_score': bins[:-1],
        'max_score': bins[1:],
        'count': bin_counts,
        'positive_count': bin_pos_counts,
        'positive_rate': bin_pos_rate
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot distribution
    ax1.bar(range(n_bins), bin_counts, alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Plot positive rate
    ax2.bar(range(n_bins), bin_pos_rate, color='r', alpha=0.7)
    ax2.set_ylim([0, max(1.0, np.max(bin_pos_rate) * 1.1)])
    ax2.set_ylabel('Positive Rate')
    ax2.set_xlabel('Score Bin')
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add positive rate labels
    for i, rate in enumerate(bin_pos_rate):
        if not np.isnan(rate) and rate > 0:
            ax2.annotate(f'{rate:.2%}', 
                         xy=(i, rate), 
                         xytext=(0, 5),
                         textcoords='offset points',
                         ha='center')
    
    plt.tight_layout()
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'score_distribution.png')
        plt.savefig(output_path)
    
    plt.close()
    
    return bin_df


def plot_confusion_matrix(
    cm: np.ndarray,
    threshold: float,
    filepath: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        threshold: Classification threshold
        filepath: Output file path
    """
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics text
    metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nSpecificity: {specificity:.4f}\nF1 Score: {f1:.4f}'
    plt.figtext(0.7, 0.25, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save if filepath provided
    if filepath:
        plt.savefig(filepath)
    else:
        output_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
        plt.savefig(output_path)
    
    plt.close() 