#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature stability analysis functionality.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import MODEL_DIR
from src.utils import configure_fonts_for_plots

# Call the font configuration function at module load time
configure_fonts_for_plots()

def analyze_feature_stability(
    df: pd.DataFrame, 
    time_column: str = 'recall_date', 
    n_bins: int = 5,
    n_jobs: int = -1
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze feature stability using Population Stability Index (PSI).
    
    Args:
        df: DataFrame with features and time column
        time_column: Time column name
        n_bins: Number of time bins
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary with PSI values for each feature
    """
    start_time = time.time()
    df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d')
    df = df.sort_values(by=time_column)
    
    df['time_bin'] = pd.cut(df[time_column], bins=n_bins, labels=False)
    
    # Exclude non-feature columns
    exclude_cols = ['input_key', time_column, 'time_bin', 'label_register', 'label_apply', 'label_approve']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Calculate PSI in parallel
    psi_results = {}
    n_features = len(feature_cols)
    
    print(f"开始并行计算特征稳定性 (PSI)，共 {n_features} 个特征...")
    
    # Worker function to calculate PSI for a single feature
    def calculate_feature_psi(feature):
        try:
            psi_values = []
            base_data = df[df['time_bin'] == 0][feature]
            
            base_hist, base_edges = np.histogram(base_data, bins=10, range=(df[feature].min(), df[feature].max()))
            base_hist = base_hist / len(base_data)
            base_hist = np.where(base_hist == 0, 0.0001, base_hist)
            
            for bin_idx in range(1, n_bins):
                curr_data = df[df['time_bin'] == bin_idx][feature]
                
                # Use same bin edges for the current distribution
                curr_hist, _ = np.histogram(curr_data, bins=base_edges)
                curr_hist = curr_hist / len(curr_data)
                curr_hist = np.where(curr_hist == 0, 0.0001, curr_hist)
                
                psi = np.sum((curr_hist - base_hist) * np.log(curr_hist / base_hist))
                psi_values.append(psi)
            
            return {
                'feature': feature,
                'psi_values': psi_values,
                'avg_psi': np.mean(psi_values),
                'max_psi': np.max(psi_values)
            }
        except Exception as e:
            print(f"\n计算特征 {feature} 的 PSI 时出错: {str(e)[:100]}")
            return {
                'feature': feature,
                'psi_values': [],
                'avg_psi': float('nan'),
                'max_psi': float('nan')
            }
    
    # Process features in parallel
    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit tasks to thread pool
        future_to_feature = {executor.submit(calculate_feature_psi, feature): feature for feature in feature_cols}
        
        # Show progress
        for future in tqdm(as_completed(future_to_feature), total=len(feature_cols), desc="计算特征稳定性"):
            feature = future_to_feature[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'\n{feature} 生成异常: {exc}')
    
    # Convert results to dictionary
    for result in results:
        feature = result['feature']
        psi_results[feature] = {
            'psi_values': result['psi_values'],
            'avg_psi': result['avg_psi'],
            'max_psi': result['max_psi']
        }
    
    # Plot PSI values
    plt.figure(figsize=(12, 8))
    avg_psi_values = [psi_results[feature]['avg_psi'] for feature in feature_cols]
    plt.barh(feature_cols, avg_psi_values)
    plt.axvline(x=0.1, color='r', linestyle='--', label='PSI=0.1 (轻微变化)')
    plt.axvline(x=0.25, color='orange', linestyle='--', label='PSI=0.25 (中等变化)')
    plt.xlabel('平均PSI值')
    plt.ylabel('特征')
    plt.title('特征稳定性分析 (PSI)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_stability_psi.png'), dpi=300)
    
    print(f"\n特征稳定性分析结果 (耗时: {time.time() - start_time:.2f}秒):")
    for feature in feature_cols:
        avg_psi = psi_results[feature]['avg_psi']
        stability = "稳定" if avg_psi < 0.1 else "轻微变化" if avg_psi < 0.25 else "显著变化"
        print(f"{feature}: 平均PSI={avg_psi:.4f} - {stability}")
    
    # Export PSI results
    psi_df = pd.DataFrame({
        'feature': list(psi_results.keys()),
        'avg_psi': [psi_results[f]['avg_psi'] for f in psi_results],
        'max_psi': [psi_results[f]['max_psi'] for f in psi_results]
    })
    psi_df = psi_df.sort_values('avg_psi')
    psi_file = os.path.join(MODEL_DIR, "feature_stability_psi.csv")
    psi_df.to_csv(psi_file, index=False)
    
    return psi_results


def plot_feature_drift(
    df: pd.DataFrame,
    feature: str,
    time_column: str = 'recall_date',
    n_bins: int = 5
) -> None:
    """
    Plot feature drift over time for a specific feature.
    
    Args:
        df: DataFrame with features and time column
        feature: Feature to plot
        time_column: Time column name
        n_bins: Number of time bins
    """
    # Convert date column and sort
    df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d')
    df = df.sort_values(by=time_column)
    
    # Create time bins
    df['time_bin'] = pd.cut(df[time_column], bins=n_bins)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Calculate feature distribution for each time bin
    grouped = df.groupby('time_bin')
    bins = 20
    
    # Get global range for consistent bins
    feature_min = df[feature].min()
    feature_max = df[feature].max()
    bin_range = (feature_min, feature_max)
    
    # Plot histogram for each time bin
    for i, (name, group) in enumerate(grouped):
        plt.subplot(n_bins, 1, i+1)
        plt.hist(group[feature], bins=bins, range=bin_range, alpha=0.7, 
                 density=True, label=f'Bin {i+1}: {name}')
        plt.title(f'Time period: {name}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'feature_drift_{feature}.png'), dpi=300)
    plt.close()


def compare_feature_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    max_features: int = 10
) -> None:
    """
    Compare feature distributions between training and testing datasets.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        features: Features to compare
        max_features: Maximum number of features to plot
    """
    # Limit number of features
    if len(features) > max_features:
        features = features[:max_features]
    
    # Create figure
    n_cols = 2
    n_rows = (len(features) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = [axes[0], axes[1]]  # Convert to list for single row
    else:
        axes = [axes]  # Single subplot case
    
    # Plot distributions
    for i, feature in enumerate(features):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i]
            
            # Get feature range
            feature_min = min(train_df[feature].min(), test_df[feature].min())
            feature_max = max(train_df[feature].max(), test_df[feature].max())
            bin_range = (feature_min, feature_max)
            
            # Plot histograms
            ax.hist(train_df[feature], bins=20, range=bin_range, alpha=0.5, 
                    density=True, label='Train')
            ax.hist(test_df[feature], bins=20, range=bin_range, alpha=0.5, 
                    density=True, label='Test')
            
            ax.set_title(f'Feature: {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'train_test_distributions.png'), dpi=300)
    plt.close() 