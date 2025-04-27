#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization functions for feature selection results.
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.core.config import MODEL_DIR
from src.utils import configure_fonts_for_plots

# Configure fonts at module level to ensure it applies to all plots
configure_fonts_for_plots()

def plot_feature_selection_summary(
    feature_scores: Dict[str, Dict[str, Any]],
    top_n: int = 30,
    filepath: Optional[str] = None
) -> None:
    """
    Plot summary of feature selection scores showing component contributions.
    
    Args:
        feature_scores: Dictionary with feature scores from FeatureSelector
        top_n: Number of top features to display
        filepath: Path to save the figure
    """
    # Set up figure
    plt.figure(figsize=(14, 10))
    
    # Extract data for plotting
    features = []
    importance_scores = []
    psi_scores = []
    stats_scores = []
    total_scores = []
    filter_reasons = []
    
    # Sort features by total score
    sorted_features = sorted(
        feature_scores.items(), 
        key=lambda x: x[1].get('score', 0), 
        reverse=True
    )
    
    # Take top N features
    for feature, scores in sorted_features[:top_n]:
        features.append(feature)
        importance_scores.append(scores.get('importance', 0) * 0.4)  # Assuming 0.4 weight
        psi_scores.append(scores.get('psi', 0) * 0.2)  # Assuming 0.2 weight
        # Combine stats scores (missing, variance, iv) with their weights
        stats_score = scores.get('stats', 0) * 0.4  # Assuming 0.4 total weight for stats
        stats_scores.append(stats_score)
        total_scores.append(scores.get('score', 0))
        filter_reasons.append(scores.get('filter_reason', None))
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'feature': features,
        'Importance (40%)': importance_scores,
        'Stability (20%)': psi_scores,
        'Statistics (40%)': stats_scores,
        'total_score': total_scores,
        'filter_reason': filter_reasons
    })
    
    # Reverse order for better visualization
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    
    # Plot stacked bars
    ax = plt.subplot(111)
    bar_width = 0.6
    
    # Plot each component
    bottom = np.zeros(len(plot_df))
    for component in ['Importance (40%)', 'Stability (20%)', 'Statistics (40%)']:
        ax.barh(plot_df['feature'], plot_df[component], bar_width, 
                left=bottom, label=component, alpha=0.8)
        bottom += plot_df[component]
    
    # Add markers for filtered features
    for i, reason in enumerate(plot_df['filter_reason']):
        if reason:
            ax.text(plot_df['total_score'].iloc[i] + 0.01, i, '! ' + reason,
                    va='center', fontsize=9, alpha=0.7, color='red')
    
    # Customize plot
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('综合得分 (Weighted Score)')
    ax.set_ylabel('特征 (Features)')
    ax.set_title('特征选择得分明细 (Feature Selection Score Components)')
    ax.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        output_path = os.path.join(MODEL_DIR, 'feature_selection_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_feature_selection_heatmap(
    feature_scores: Dict[str, Dict[str, Any]],
    top_n: int = 30,
    filepath: Optional[str] = None
) -> None:
    """
    Create a heatmap showing feature selection metrics for top features.
    
    Args:
        feature_scores: Dictionary with feature scores from FeatureSelector
        top_n: Number of top features to display
        filepath: Path to save the figure
    """
    # Set up figure
    plt.figure(figsize=(16, 12))
    
    # Sort features by total score
    sorted_features = sorted(
        feature_scores.items(), 
        key=lambda x: x[1].get('score', 0), 
        reverse=True
    )
    
    # Extract data for the top features
    features = []
    metrics = []
    
    for feature, scores in sorted_features[:top_n]:
        features.append(feature)
        
        # Extract metrics with proper handling for missing values
        feature_metrics = {
            'Importance': scores.get('importance', 0),
            'Stability': scores.get('psi', 0),
            'Missing Rate': 1.0 - scores.get('missing_rate', 0),  # Invert for visual consistency
            'Variance': scores.get('variance', 0),
            'IV': scores.get('iv', 0),
            'Target Correlation': abs(scores.get('correlation_with_target', 0)),
            'Total Score': scores.get('score', 0)
        }
        metrics.append(feature_metrics)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(metrics, index=features)
    
    # Normalize columns to 0-1 range for better visualization
    for col in heatmap_df.columns:
        if col != 'Total Score':  # Keep total score as is
            col_min = heatmap_df[col].min()
            col_max = heatmap_df[col].max()
            if col_max > col_min:
                heatmap_df[col] = (heatmap_df[col] - col_min) / (col_max - col_min)
    
    # Create a custom colormap that goes from red to green
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_RdYlGn', ['#FF5A5F', '#FFEDA0', '#91CF60'], N=100
    )
    
    # Plot heatmap
    ax = sns.heatmap(
        heatmap_df, 
        annot=True,
        cmap=custom_cmap,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': '归一化分数 (Normalized Score)'}
    )
    
    # Customize plot
    plt.title('特征选择指标热图 (Feature Selection Metrics Heatmap)')
    plt.tight_layout()
    
    # Save figure
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        output_path = os.path.join(MODEL_DIR, 'feature_selection_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_feature_correlation_network(
    train_df: pd.DataFrame,
    feature_scores: Dict[str, Dict[str, Any]],
    top_n: int = 30,
    correlation_threshold: float = 0.7,
    filepath: Optional[str] = None
) -> None:
    """
    Plot a network graph of feature correlations.
    
    Args:
        train_df: Training DataFrame
        feature_scores: Dictionary with feature scores
        top_n: Number of top features to include
        correlation_threshold: Minimum correlation for drawing connections
        filepath: Path to save the figure
    """
    try:
        import networkx as nx
        
        # Set up figure
        plt.figure(figsize=(16, 12))
        
        # Get top features by score
        top_features = [
            feature for feature, _ in sorted(
                feature_scores.items(), 
                key=lambda x: x[1].get('score', 0), 
                reverse=True
            )[:top_n]
        ]
        
        # Filter features available in dataframe
        available_features = [f for f in top_features if f in train_df.columns]
        
        if len(available_features) < 2:
            print("Too few features for correlation network plot")
            return
        
        # Calculate correlation matrix
        corr_matrix = train_df[available_features].corr().abs()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for feature in available_features:
            score = feature_scores.get(feature, {}).get('score', 0)
            importance = feature_scores.get(feature, {}).get('importance', 0)
            G.add_node(
                feature, 
                score=score,
                importance=importance
            )
        
        # Add edges for correlated features
        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features):
                if i < j:  # Avoid duplicate edges
                    corr = corr_matrix.loc[feat1, feat2]
                    if corr >= correlation_threshold:
                        G.add_edge(feat1, feat2, weight=corr)
        
        # Calculate node positions using spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Get scores for node size and color
        scores = [G.nodes[n]['score'] * 500 for n in G.nodes()]
        importances = [G.nodes[n]['importance'] for n in G.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_size=scores,
            node_color=importances,
            cmap='viridis',
            alpha=0.7
        )
        
        # Create color bar for importance
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(importances)
        cbar = plt.colorbar(sm, label='Feature Importance')
        
        # Draw edges with varying width based on correlation
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=[(u, v)], 
                width=d['weight'] * 2,
                alpha=d['weight'],
                edge_color='grey'
            )
        
        # Draw labels with varying sizes
        label_sizes = {n: 8 + G.nodes[n]['score'] * 4 for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, 
            font_size=10,
            font_family='sans-serif',
            font_weight='bold'
        )
        
        # Customize plot
        plt.title('特征相关性网络图 (Feature Correlation Network)')
        plt.axis('off')
        plt.tight_layout()
        
        # Add annotation for correlation threshold
        plt.figtext(
            0.02, 0.02, 
            f"仅显示相关性 ≥ {correlation_threshold} 的连接\n节点大小代表特征得分，颜色深浅代表特征重要性",
            fontsize=10
        )
        
        # Save figure
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            output_path = os.path.join(MODEL_DIR, 'feature_correlation_network.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        print("networkx package is required for correlation network visualization")


def plot_feature_selection_stats_distribution(
    feature_stats: pd.DataFrame,
    feature_scores: Dict[str, Dict[str, Any]],
    selected_features: List[str],
    filepath: Optional[str] = None
) -> None:
    """
    Plot distributions of feature statistics, highlighting selected vs. rejected features.
    
    Args:
        feature_stats: DataFrame with feature statistics
        feature_scores: Dictionary with feature scores
        selected_features: List of selected features
        filepath: Path to save the figure
    """
    # Set feature selection status
    feature_stats['selected'] = feature_stats['feature'].isin(selected_features)
    
    # Create plot with multiple panels
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot IV distribution
    sns.histplot(
        data=feature_stats, x='iv', hue='selected',
        bins=20, alpha=0.6, ax=axs[0, 0]
    )
    axs[0, 0].set_title('IV值分布 (Information Value)')
    axs[0, 0].set_xlabel('IV值')
    axs[0, 0].axvline(0.02, color='red', linestyle='--', label='最小IV阈值')
    axs[0, 0].legend(labels=['最小IV阈值', '未选中特征', '选中特征'])
    
    # Plot missing rate distribution
    sns.histplot(
        data=feature_stats, x='missing_or_zero_rate', hue='selected',
        bins=20, alpha=0.6, ax=axs[0, 1]
    )
    axs[0, 1].set_title('缺失率分布 (Missing Rate)')
    axs[0, 1].set_xlabel('缺失率')
    axs[0, 1].axvline(0.99, color='red', linestyle='--', label='最大缺失率阈值')
    axs[0, 1].legend(labels=['最大缺失率阈值', '未选中特征', '选中特征'])
    
    # Plot correlation with target distribution
    sns.histplot(
        data=feature_stats, x='correlation_with_target', hue='selected',
        bins=20, alpha=0.6, ax=axs[1, 0]
    )
    axs[1, 0].set_title('目标相关性分布 (Target Correlation)')
    axs[1, 0].set_xlabel('目标相关性')
    axs[1, 0].legend(labels=['未选中特征', '选中特征'])
    
    # Plot variance distribution
    sns.histplot(
        data=feature_stats, x='variance', hue='selected',
        bins=20, alpha=0.6, ax=axs[1, 1], log_scale=(True, False)
    )
    axs[1, 1].set_title('方差分布 (Variance)')
    axs[1, 1].set_xlabel('特征方差 (对数刻度)')
    if 'min_variance' in feature_scores:
        min_var = feature_scores['min_variance']
        axs[1, 1].axvline(min_var, color='red', linestyle='--', label='最小方差阈值')
        axs[1, 1].legend(labels=['最小方差阈值', '未选中特征', '选中特征'])
    else:
        axs[1, 1].legend(labels=['未选中特征', '选中特征'])
    
    # Add overall title
    plt.suptitle('特征统计分布对比 (Feature Statistics Distributions)', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        output_path = os.path.join(MODEL_DIR, 'feature_selection_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_feature_selection_results(
    feature_scores: Dict[str, Dict[str, Any]],
    feature_stats: pd.DataFrame,
    selected_features: List[str],
    train_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Generate comprehensive visualization for feature selection results.
    
    Args:
        feature_scores: Dictionary with feature scores
        feature_stats: DataFrame with feature statistics
        selected_features: List of selected features
        train_df: Training DataFrame (for correlation analysis)
        output_dir: Directory to save visualizations
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, 'feature_selection_viz')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"正在生成特征选择可视化结果，将保存至: {output_dir}")
    
    # 1. Feature selection summary barplot
    print("生成特征选择得分明细图...")
    plot_feature_selection_summary(
        feature_scores,
        top_n=30,
        filepath=os.path.join(output_dir, 'feature_selection_summary.png')
    )
    
    # 2. Feature metrics heatmap
    print("生成特征选择指标热图...")
    plot_feature_selection_heatmap(
        feature_scores,
        top_n=30,
        filepath=os.path.join(output_dir, 'feature_selection_heatmap.png')
    )
    
    # 3. Feature correlation network (if train_df is provided)
    if train_df is not None:
        print("生成特征相关性网络图...")
        plot_feature_correlation_network(
            train_df,
            feature_scores,
            top_n=30,
            correlation_threshold=0.7,
            filepath=os.path.join(output_dir, 'feature_correlation_network.png')
        )
    
    # 4. Feature stats distributions
    print("生成特征统计分布对比图...")
    plot_feature_selection_stats_distribution(
        feature_stats,
        feature_scores,
        selected_features,
        filepath=os.path.join(output_dir, 'feature_selection_distributions.png')
    )
    
    print(f"特征选择可视化完成，共生成 {3 if train_df is None else 4} 张图表。") 