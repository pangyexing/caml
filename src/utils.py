#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility module for ML pipeline.
Contains shared utility functions, constants, and helper functions.
"""

import os
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Common column constants
ESSENTIAL_COLS = ['input_key', 'recall_date']
LABEL_COLS = ['label_register', 'label_apply', 'label_approve']
EXCLUDE_COLS = ESSENTIAL_COLS + LABEL_COLS + ['time_bin', 'score', 'customer_name', 'speaking_duration', 'label_intention']

# Directory constants
MODEL_DIR = "funnel_models"
TUNING_DIR = "tuned_models"
RESULTS_DIR = "optimization_results"

def ensure_dirs():
    """Ensure all required directories exist."""
    for directory in [MODEL_DIR, TUNING_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"Created necessary directories: {', '.join([MODEL_DIR, TUNING_DIR, RESULTS_DIR])}")

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Get feature columns from DataFrame by excluding non-feature columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    return [col for col in df.columns if col not in EXCLUDE_COLS]

def load_feature_importance(importance_file: str) -> Dict[str, float]:
    """
    Load feature importance from CSV file.
    
    Args:
        importance_file: Path to importance CSV or text file
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        importance_df = pd.read_csv(importance_file)
        
        importance_col = 'importance' if 'importance' in importance_df.columns else 'gain'
        
        importance_df = importance_df.sort_values(by=importance_col, ascending=False)
        
        return dict(zip(importance_df['feature'], importance_df[importance_col]))
    except:
        # Try loading as text file with one feature per line
        with open(importance_file, 'r') as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        return {feature: len(features) - i for i, feature in enumerate(features)}

def load_feature_list(feature_file: str) -> List[str]:
    """
    Load feature list from text file.
    
    Args:
        feature_file: Path to feature list file
        
    Returns:
        List of feature names
    """
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    return features

def save_feature_list(features: List[str], output_file: str) -> None:
    """
    Save feature list to text file.
    
    Args:
        features: List of feature names
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"特征列表已保存至 {output_file}")

def save_model_metrics(metrics: Dict[str, Any], model_name: str, output_file: str) -> None:
    """
    Save model metrics to file.
    
    Args:
        metrics: Model metrics dictionary
        model_name: Model name
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("="*50 + "\n\n")
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
            elif isinstance(value, dict):
                f.write(f"{metric}:\n")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        f.write(f"  {k}: {v:.4f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{metric}: {value}\n")
    
    print(f"模型指标已保存至 {output_file}")

def timer(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return result
    return wrapper

def analyze_label_distribution(df: pd.DataFrame, target: str = 'label_apply') -> Dict[str, Any]:
    """
    Analyze target label distribution.
    
    Args:
        df: Input DataFrame
        target: Target column name
        
    Returns:
        Distribution statistics
    """
    pos_count = df[target].sum()
    total_count = len(df)
    pos_ratio = pos_count / total_count
    
    print(f"标签分布分析: {target}")
    print(f"  正样本数量: {pos_count}")
    print(f"  总样本数量: {total_count}")
    print(f"  正样本比例: {pos_ratio:.2%}")
    
    # Visualize label distribution
    plt.figure(figsize=(8, 6))
    df[target].value_counts().plot(kind='bar')
    plt.title(f'标签分布: {target}')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.xticks([0, 1], ['负样本', '正样本'])
    plt.tight_layout()
    plt.savefig(f'{target}_distribution.png')
    plt.close()
    
    return {
        'positive': pos_count,
        'total': total_count,
        'ratio': pos_ratio
    }

def calculate_improvement(initial_metrics: Dict[str, float], 
                         final_metrics: Dict[str, float], 
                         metrics_list: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate improvement between initial and final metrics.
    
    Args:
        initial_metrics: Initial metrics dictionary
        final_metrics: Final metrics dictionary
        metrics_list: List of metrics to calculate improvement for (default: all matching metrics)
        
    Returns:
        Dictionary with improvement metrics
    """
    if metrics_list is None:
        metrics_list = [m for m in initial_metrics.keys() if m in final_metrics and isinstance(initial_metrics[m], (int, float))]
    
    improvements = {}
    for metric in metrics_list:
        if metric in initial_metrics and metric in final_metrics:
            initial_val = initial_metrics[metric]
            final_val = final_metrics[metric]
            
            if initial_val != 0:
                abs_improvement = final_val - initial_val
                rel_improvement = (abs_improvement / initial_val) * 100
                
                improvements[metric] = {
                    'initial': initial_val,
                    'final': final_val,
                    'absolute': abs_improvement,
                    'relative': rel_improvement
                }
    
    return improvements

def plot_metrics_comparison(initial_metrics: Dict[str, float], 
                           final_metrics: Dict[str, float], 
                           title: str = "Model Metrics Comparison", 
                           output_file: str = "metrics_comparison.png") -> None:
    """
    Plot comparison of metrics between initial and final models.
    
    Args:
        initial_metrics: Initial metrics dictionary
        final_metrics: Final metrics dictionary
        title: Plot title
        output_file: Output file path
    """
    metrics = [m for m in initial_metrics.keys() if m in final_metrics 
              and isinstance(initial_metrics[m], (int, float))
              and m not in ['best_threshold', 'f1_threshold']]
    
    initial_values = [initial_metrics[m] for m in metrics]
    final_values = [final_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, initial_values, width, label='Initial Model')
    rects2 = ax.bar(x + width/2, final_values, width, label='Final Model')
    
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"指标对比图已保存至 {output_file}")

def export_model_summary(model_info: Dict[str, Any], output_file: str) -> None:
    """
    Export model training summary to file.
    
    Args:
        model_info: Model information dictionary
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("模型训练总结\n")
        f.write("="*50 + "\n\n")
        
        # Feature information
        if 'final_features' in model_info:
            f.write(f"特征数量: {len(model_info['final_features'])}\n")
        
        # Metrics information
        if 'initial_metrics' in model_info and 'final_metrics' in model_info:
            f.write("\n初始模型指标:\n")
            for metric, value in model_info['initial_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n最终模型指标:\n")
            for metric, value in model_info['final_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # Calculate improvements
            improvements = calculate_improvement(model_info['initial_metrics'], model_info['final_metrics'])
            
            f.write("\n模型改进:\n")
            for metric, values in improvements.items():
                f.write(f"  {metric}: {values['absolute']:.4f} ({values['relative']:.2f}%)\n")
        
        # Model and feature file paths
        if 'model_file' in model_info:
            f.write(f"\n模型文件: {model_info['model_file']}\n")
        
        if 'feature_file' in model_info:
            f.write(f"特征列表文件: {model_info['feature_file']}\n")
    
    print(f"模型训练总结已保存至 {output_file}") 