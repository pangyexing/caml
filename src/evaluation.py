#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation module.
Shared functions for model evaluation, score binning, and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    recall_score,
    precision_score,
    f1_score,
    roc_curve
)

def create_score_bins(df: pd.DataFrame, 
                     y_true: pd.Series = None,
                     score_col: str = 'score', 
                     target_col: str = None,
                     n_bins: int = 10,
                     output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create score bins and analyze performance in each bin.
    
    Args:
        df: DataFrame with scores
        y_true: True labels (optional, if not provided will use target_col from df)
        score_col: Score column name
        target_col: Target column name (used if y_true not provided)
        n_bins: Number of bins
        output_file: Output file path (optional)
        
    Returns:
        DataFrame with bin statistics
    """
    # Use either provided y_true or extract from df using target_col
    if y_true is None and target_col:
        y_true = df[target_col]
    
    # Create score bins
    score_bins_cat = pd.qcut(df[score_col], n_bins, duplicates='drop', retbins=False)
    
    # Calculate statistics for each bin
    grouped = df.groupby(score_bins_cat)
    counts = grouped.size()
    
    if y_true is not None:
        # Make sure y_true has the same index as df
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true, index=df.index)
        elif y_true.index.equals(df.index) is False:
            y_true = y_true.reset_index(drop=True)
            
        df_with_labels = df.copy()
        df_with_labels['true_label'] = y_true
        grouped = df_with_labels.groupby(score_bins_cat)
        events = grouped['true_label'].sum()
    else:
        events = pd.Series(0, index=counts.index)
    
    non_events = counts - events
    
    bin_stats_interval_idx = pd.DataFrame({
        'Count': counts,
        'Non-event': non_events,
        'Event': events
    }).sort_index()
    
    # Convert intervals to string representation
    bin_stats_interval_idx['Bin'] = [f"[{interval.left:.1f}, {interval.right:.1f})" 
                                    for interval in bin_stats_interval_idx.index]
    
    # Reset index and reorder columns
    bin_stats = bin_stats_interval_idx.reset_index(drop=True)[['Bin', 'Count', 'Non-event', 'Event']]
    
    # Calculate derived metrics
    bin_stats['Event rate'] = (bin_stats['Event'] / bin_stats['Count']).fillna(0)
    
    total_event = bin_stats['Event'].sum()
    total_non_event = bin_stats['Non-event'].sum()
    epsilon = 1e-10
    
    bin_stats['p_event_i'] = bin_stats['Event'] / (total_event + epsilon)
    bin_stats['p_non_event_i'] = bin_stats['Non-event'] / (total_non_event + epsilon)
    
    bin_stats['WoE'] = np.log(np.maximum(bin_stats['p_non_event_i'], epsilon) / 
                             np.maximum(bin_stats['p_event_i'], epsilon))
    bin_stats['IV'] = (bin_stats['p_non_event_i'] - bin_stats['p_event_i']) * bin_stats['WoE']
    
    bin_stats['cum_event'] = bin_stats['Event'].cumsum() / (total_event + epsilon)
    bin_stats['cum_non_event'] = bin_stats['Non-event'].cumsum() / (total_non_event + epsilon)
    bin_stats['KS'] = abs(bin_stats['cum_event'] - bin_stats['cum_non_event'])
    
    bin_stats = bin_stats.drop(['p_event_i', 'p_non_event_i'], axis=1)
    
    # Save to file if output_file is provided
    if output_file:
        bin_stats.to_csv(output_file, index=False)
        print(f"分数分箱已保存至 {output_file}")
    
    return bin_stats

def plot_score_distribution(bin_stats: pd.DataFrame, 
                           target: str, 
                           output_file: Optional[str] = None) -> None:
    """
    Plot score distribution and KS curve.
    
    Args:
        bin_stats: Bin statistics DataFrame
        target: Target variable name (for title)
        output_file: Output file path (if None, use default)
    """
    if output_file is None:
        output_file = f"{target}_score_distribution.png"
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    ax1.bar(bin_stats.index, bin_stats['Event rate'], color='skyblue')
    ax1.set_ylabel('Event Rate', color='blue')
    ax1.set_xlabel('Score Bin')
    ax1.set_xticks(bin_stats.index)
    ax1.set_xticklabels(bin_stats['Bin'], rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(bin_stats.index, bin_stats['KS'], 'r-', marker='o')
    ax2.set_ylabel('KS Value', color='red')

    plt.title(f'Score Distribution and KS for {target}')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"分数分布图已保存至 {output_file}")

def evaluate_predictions(y_true: pd.Series, 
                        y_pred_proba: pd.Series, 
                        threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Evaluate prediction results.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (if None, find optimal F1)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate ROC AUC
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate PR AUC
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold if not provided
    if threshold is None:
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
        threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5
    
    # Calculate metrics at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    
    # Calculate KS statistic
    # Sort predictions and corresponding true values
    sorted_indices = np.argsort(y_pred_proba)
    sorted_y_true = y_true.iloc[sorted_indices] if isinstance(y_true, pd.Series) else y_true[sorted_indices]
    
    # Calculate cumulative distributions
    n_pos = np.sum(sorted_y_true == 1)
    n_neg = len(sorted_y_true) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(sorted_y_true == 1) / n_pos
        cum_neg = np.cumsum(sorted_y_true == 0) / n_neg
        ks_value = np.max(np.abs(cum_pos - cum_neg))
    else:
        ks_value = 0.0
    
    # Collect metrics
    metrics = {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'ks': ks_value,
        'f1': f1,
        'recall': recall_val,
        'precision': precision_val,
        'threshold': threshold
    }
    
    return metrics

def plot_precision_recall_curve(y_true: pd.Series, 
                               y_pred_proba: pd.Series, 
                               target: str, 
                               output_file: Optional[str] = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target: Target variable name (for title)
        output_file: Output file path (if None, use default)
    """
    if output_file is None:
        output_file = f"{target}_precision_recall_curve.png"
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    plt.grid(True)
    
    # Add threshold annotations
    thresholds_to_annotate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds_to_annotate:
        idx = (np.abs(thresholds - threshold)).argmin() if len(thresholds) > 0 else 0
        if idx < len(precision) - 1:  # Ensure index is valid
            plt.annotate(f'{threshold:.1f}', 
                        xy=(recall[idx], precision[idx]),
                        xytext=(recall[idx] - 0.05, precision[idx] - 0.05),
                        arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"PR曲线已保存至 {output_file}") 