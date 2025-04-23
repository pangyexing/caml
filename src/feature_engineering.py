#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module.
Handles feature selection, stability analysis, and importance evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def analyze_feature_stability(df: pd.DataFrame, 
                              time_column: str = 'recall_date', 
                              n_bins: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Analyze feature stability using Population Stability Index (PSI).
    
    Args:
        df: DataFrame with features and time column
        time_column: Time column name
        n_bins: Number of time bins
    
    Returns:
        Dictionary with PSI values for each feature
    """
    df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d')
    df = df.sort_values(by=time_column)
    
    df['time_bin'] = pd.cut(df[time_column], bins=n_bins, labels=False)
    
    # Exclude non-feature columns
    exclude_cols = ['input_key', time_column, 'time_bin', 'label_register', 'label_apply', 'label_approve']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Calculate PSI
    psi_results = {}
    n_features = len(feature_cols)
    
    for i, feature in enumerate(feature_cols):
        print(f"\rProcessing feature {i+1}/{n_features}: {feature}{' '*20}", end='')
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
        
        psi_results[feature] = {
            'psi_values': psi_values,
            'avg_psi': np.mean(psi_values),
            'max_psi': np.max(psi_values)
        }
    print()
    
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
    plt.savefig('feature_stability_psi.png')
    
    print("\n特征稳定性分析结果:")
    for feature in feature_cols:
        avg_psi = psi_results[feature]['avg_psi']
        stability = "稳定" if avg_psi < 0.1 else "轻微变化" if avg_psi < 0.25 else "显著变化"
        print(f"{feature}: 平均PSI={avg_psi:.4f} - {stability}")
    
    return psi_results

def analyze_features_for_selection_parallel(train_df: pd.DataFrame, 
                                           feature_cols: List[str], 
                                           target: str = 'label_apply',
                                           n_jobs: int = 4) -> pd.DataFrame:
    """
    Analyze features for selection using parallel processing.
    
    Args:
        train_df: Training DataFrame
        feature_cols: Feature columns to analyze
        target: Target variable name
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with feature statistics
    """
    print(f"开始并行分析特征选择指标，共 {len(feature_cols)} 个特征...")
    start_time = time.time()
    
    # Calculate basic statistics for each feature
    def analyze_feature(feature):
        try:
            # Missing or zero rate
            missing_rate = (train_df[feature].isna() | (train_df[feature] == 0.0)).mean()
            
            # Variance
            variance = train_df[feature].var()
            
            # Correlation with target
            correlation = train_df[feature].corr(train_df[target])
            
            # Calculate IV value
            try:
                # Bin the feature
                bins = 10
                if train_df[feature].nunique() <= bins:
                    binned_feature = train_df[feature]
                else:
                    binned_feature = pd.qcut(train_df[feature], bins, duplicates='drop')
                
                # Calculate event and non-event counts for each bin
                grouped = train_df.groupby(binned_feature)
                counts = grouped.size()
                events = grouped[target].sum()
                non_events = counts - events
                
                # Calculate IV
                total_events = events.sum()
                total_non_events = non_events.sum()
                
                # Avoid division by zero
                epsilon = 1e-10
                p_event = (events / total_events).fillna(0)
                p_non_event = (non_events / total_non_events).fillna(0)
                
                woe = np.log(np.maximum(p_non_event, epsilon) / np.maximum(p_event, epsilon))
                iv = ((p_non_event - p_event) * woe).sum()
            except Exception as e:
                iv = np.nan
                print(f"计算特征 {feature} 的IV值时出错: {str(e)[:100]}")
            
            return {
                'feature': feature,
                'missing_or_zero_rate': missing_rate,
                'variance': variance,
                'correlation_with_target': correlation,
                'iv': iv
            }
        except Exception as e:
            print(f"分析特征 {feature} 时出错: {str(e)[:100]}")
            return {
                'feature': feature,
                'missing_or_zero_rate': 1.0,  # Assume all missing
                'variance': 0.0,
                'correlation_with_target': 0.0,
                'iv': np.nan
            }
    
    # Process features in parallel
    stats_data = []
    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        # Submit tasks to thread pool
        future_to_feature = {executor.submit(analyze_feature, feature): feature for feature in feature_cols}
        
        # Show progress
        for future in tqdm(as_completed(future_to_feature), total=len(feature_cols)):
            feature = future_to_feature[future]
            try:
                result = future.result()
                stats_data.append(result)
            except Exception as exc:
                print(f'{feature} 生成异常: {exc}')
    
    # Create feature statistics DataFrame
    feature_stats = pd.DataFrame(stats_data)
    
    # Add additional summary statistics
    if not feature_stats.empty:
        # Calculate feature distribution statistics
        for col in ['missing_or_zero_rate', 'variance', 'correlation_with_target', 'iv']:
            if col in feature_stats.columns:
                valid_values = feature_stats[col].dropna()
                if not valid_values.empty:
                    print(f"\n{col} 分布统计:")
                    print(f"  最小值: {valid_values.min():.4f}")
                    print(f"  25分位: {valid_values.quantile(0.25):.4f}")
                    print(f"  中位数: {valid_values.median():.4f}")
                    print(f"  75分位: {valid_values.quantile(0.75):.4f}")
                    print(f"  最大值: {valid_values.max():.4f}")
    
    # Save feature statistics
    model_dir = "funnel_models"
    os.makedirs(model_dir, exist_ok=True)
    feature_stats.to_csv(os.path.join(model_dir, "feature_selection_stats.csv"), index=False)
    
    # Provide feature selection suggestions
    print(f"\n特征分析完成，耗时 {time.time() - start_time:.2f} 秒")
    print("\n特征选择建议:")
    
    # High IV features
    high_iv_features = feature_stats.sort_values('iv', ascending=False).head(20)
    print(f"\n信息量(IV)最高的20个特征:")
    for idx, row in high_iv_features.iterrows():
        print(f"  {row['feature']}: IV={row['iv']:.4f}")
    
    # High correlation features
    high_corr_features = feature_stats.sort_values('correlation_with_target', ascending=False).head(20)
    print(f"\n与目标变量相关性最高的20个特征:")
    for idx, row in high_corr_features.iterrows():
        print(f"  {row['feature']}: 相关性={row['correlation_with_target']:.4f}")
    
    # High missing rate features
    high_missing_features = feature_stats[feature_stats['missing_or_zero_rate'] > 0.99]
    if not high_missing_features.empty:
        print(f"\n缺失或为0比例高于99%的特征 (建议考虑移除):")
        for idx, row in high_missing_features.iterrows():
            print(f"  {row['feature']}: 缺失或为0比例={row['missing_or_zero_rate']:.2%}")
    
    # Low variance features
    low_var_threshold = feature_stats['variance'].quantile(0.1)
    low_var_features = feature_stats[feature_stats['variance'] < low_var_threshold]
    if not low_var_features.empty:
        print(f"\n方差较低的特征 (建议考虑移除):")
        for idx, row in low_var_features.iterrows():
            print(f"  {row['feature']}: 方差={row['variance']:.6f}")
    
    return feature_stats

class FeatureSelector:
    """Feature selection and evaluation class."""
    
    def __init__(self, weights=None, min_importance_pct=0.01, max_psi=0.25, 
                 max_missing_rate=0.99, min_variance=None, min_iv=0.02,
                 correlation_threshold=0.9, must_include=None, must_exclude=None):
        """
        Initialize feature selector
        
        Args:
            weights: Weight dictionary for scoring factors
            min_importance_pct: Minimum importance percentage threshold
            max_psi: Maximum PSI threshold
            max_missing_rate: Maximum missing rate
            min_variance: Minimum variance threshold
            min_iv: Minimum IV value threshold
            correlation_threshold: Feature correlation threshold
            must_include: Features that must be included
            must_exclude: Features that must be excluded
        """
        # Default weights
        self.default_weights = {
            'importance': 0.4,  # Model importance
            'psi': 0.2,         # Stability
            'missing': 0.1,     # Missing rate
            'variance': 0.1,    # Variance
            'iv': 0.2           # Information value
        }
        self.weights = weights if weights is not None else self.default_weights
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
        
        # Thresholds
        self.min_importance_pct = min_importance_pct
        self.max_psi = max_psi
        self.max_missing_rate = max_missing_rate
        self.min_variance = min_variance
        self.min_iv = min_iv
        self.correlation_threshold = correlation_threshold
        
        # Must include/exclude features
        self.must_include = set(must_include or [])
        self.must_exclude = set(must_exclude or [])
    
    def _normalize_score(self, value, min_val, max_val, higher_is_better=True):
        """Normalize value to 0-1 range"""
        if min_val == max_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return normalized if higher_is_better else (1 - normalized)
    
    def score_importance(self, feature, importance_dict, total_importance):
        """Score feature importance"""
        if feature not in importance_dict:
            return 0, None
        
        importance = importance_dict[feature]
        rel_importance = importance / total_importance
        
        # Flag features below threshold
        filter_reason = f"低重要性 ({rel_importance:.4f})" if rel_importance < self.min_importance_pct else None
        
        return rel_importance, filter_reason
    
    def score_psi(self, feature, psi_results):
        """Score feature stability (PSI)"""
        if feature not in psi_results:
            return 0.5, None  # Default medium stability
        
        psi_avg = psi_results[feature]['avg_psi']
        psi_score = 1.0 - min(psi_avg / self.max_psi, 1.0)
        
        # Flag unstable features
        filter_reason = f"不稳定 (PSI={psi_avg:.4f})" if psi_avg > self.max_psi else None
        
        return psi_score, filter_reason
    
    def score_stats(self, feature, stats_dict):
        """Score based on feature statistics (missing rate, variance, IV)"""
        if feature not in stats_dict:
            return 0.5, None  # Default medium score
        
        feature_stats = stats_dict[feature]
        
        # Extract statistics
        missing_rate = feature_stats.get('missing_or_zero_rate', 0)
        variance = feature_stats.get('variance', 0)
        iv = feature_stats.get('iv', 0)
        
        # Normalize scores (lower missing rate is better)
        missing_score = 1.0 - missing_rate
        
        # Normalize variance score (if threshold provided)
        variance_score = 0.5
        if self.min_variance is not None:
            variance_score = min(1.0, variance / self.min_variance)
        
        # Normalize IV score
        iv_score = 1.0 if iv >= self.min_iv else iv / self.min_iv
        
        # Combined score (weighted average)
        stat_weight_sum = self.weights.get('missing', 0.1) + self.weights.get('variance', 0.1) + self.weights.get('iv', 0.2)
        if stat_weight_sum > 0:
            weights = {
                'missing': self.weights.get('missing', 0.1) / stat_weight_sum,
                'variance': self.weights.get('variance', 0.1) / stat_weight_sum,
                'iv': self.weights.get('iv', 0.2) / stat_weight_sum
            }
            combined_score = (weights['missing'] * missing_score + 
                              weights['variance'] * variance_score + 
                              weights['iv'] * iv_score)
        else:
            combined_score = (0.25 * missing_score + 0.25 * variance_score + 0.5 * iv_score)
        
        # Check filter conditions
        filter_reason = None
        if missing_rate > self.max_missing_rate:
            filter_reason = f"高缺失率 ({missing_rate:.2%})"
        elif self.min_variance is not None and variance < self.min_variance:
            filter_reason = f"低方差 ({variance:.6f})"
        elif iv < self.min_iv:
            filter_reason = f"低IV值 ({iv:.4f})"
        
        return combined_score, filter_reason
    
    def calculate_feature_scores(self, feature_cols, importance_dict=None, psi_results=None, feature_stats=None):
        """
        Calculate comprehensive score for each feature and filter reasons
        
        Args:
            feature_cols: Feature column list
            importance_dict: Feature importance dictionary
            psi_results: PSI stability results
            feature_stats: Feature statistics DataFrame
            
        Returns:
            Dictionary of feature scores and filter reasons
        """
        feature_scores = {}
        
        # Preprocess data to reduce calculations in loop
        total_importance = sum(importance_dict.values()) if importance_dict else 0
        
        # Build stats dictionary from feature_stats DataFrame
        stats_dict = {}
        if feature_stats is not None:
            for _, row in feature_stats.iterrows():
                feature = row['feature']
                stats_dict[feature] = row
        
        # Calculate scores for all features
        for feature in feature_cols:
            # Check must exclude features
            if feature in self.must_exclude:
                feature_scores[feature] = {
                    'score': 0,
                    'filter_reason': '手动排除'
                }
                continue
                
            # Check must include features
            if feature in self.must_include:
                feature_scores[feature] = {
                    'score': 1,
                    'filter_reason': None,
                    'must_include': True
                }
                continue
            
            # Calculate model importance score
            importance_score, importance_filter = 0.5, None
            if importance_dict:
                importance_score, importance_filter = self.score_importance(feature, importance_dict, total_importance)
            
            # Calculate PSI stability score
            psi_score, psi_filter = 0.5, None
            if psi_results:
                psi_score, psi_filter = self.score_psi(feature, psi_results)
            
            # Calculate feature statistics score
            stats_score, stats_filter = 0.5, None
            if feature_stats is not None:
                stats_score, stats_filter = self.score_stats(feature, stats_dict)
            
            # Determine filter reason (priority: importance > PSI > statistics)
            filter_reason = importance_filter or psi_filter or stats_filter
            
            # Calculate combined score
            score = (self.weights.get('importance', 0.4) * importance_score + 
                     self.weights.get('psi', 0.2) * psi_score + 
                     (self.weights.get('missing', 0.1) + 
                      self.weights.get('variance', 0.1) + 
                      self.weights.get('iv', 0.2)) * stats_score)
            
            # Save feature score and details
            feature_scores[feature] = {
                'importance': importance_score,
                'psi': psi_score, 
                'stats': stats_score,
                'score': score,
                'filter_reason': filter_reason
            }
            
            # Add detailed statistics
            if feature_stats is not None and feature in stats_dict:
                feature_scores[feature]['missing_rate'] = stats_dict[feature].get('missing_or_zero_rate', 0)
                feature_scores[feature]['variance'] = stats_dict[feature].get('variance', 0)  
                feature_scores[feature]['iv'] = stats_dict[feature].get('iv', 0)
                feature_scores[feature]['correlation_with_target'] = stats_dict[feature].get('correlation_with_target', 0)
        
        return feature_scores
    
    def detect_correlated_features(self, train_df, feature_scores, top_features=None):
        """
        Detect and handle highly correlated features
        
        Args:
            train_df: Training DataFrame
            feature_scores: Feature scores dictionary
            top_features: Pre-calculated top features by importance
            
        Returns:
            Updated feature_scores
        """
        # Get valid features (score > 0 and not filtered)
        valid_features = [f for f, info in feature_scores.items() 
                        if info.get('score', 0) > 0 and not info.get('filter_reason')]
        
        if len(valid_features) <= 1:
            return feature_scores
        
        # If too many features, only check top features
        max_check = min(500, len(valid_features))
        
        if top_features:
            check_features = [f for f in top_features if f in valid_features][:max_check]
        else:
            # Sort by score
            check_features = sorted(valid_features, 
                                  key=lambda f: feature_scores[f]['score'], 
                                  reverse=True)[:max_check]
        
        print(f"开始检查 {len(check_features)} 个特征的相关性...")
        
        # Extract these features from dataset
        feature_df = train_df[check_features].copy()
        
        # Calculate correlation matrix in batches
        batch_size = 100
        correlated_pairs = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(check_features), batch_size):
            batch_end = min(i + batch_size, len(check_features))
            batch_features = check_features[i:batch_end]
            
            batch_corr = feature_df[batch_features].corr().abs()
            
            # Extract highly correlated feature pairs
            for idx, feature1 in enumerate(batch_features):
                for feature2 in batch_features[idx+1:]:
                    correlation = batch_corr.loc[feature1, feature2]
                    if correlation >= self.correlation_threshold:
                        correlated_pairs.append((feature1, feature2, correlation))
        
        # Handle correlated features
        if correlated_pairs:
            print(f"发现 {len(correlated_pairs)} 对高度相关的特征 (相关性 >= {self.correlation_threshold})")
            
            # Sort by correlation (descending)
            correlated_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # For each correlated pair, keep the one with higher score
            features_to_filter = set()
            for feature1, feature2, correlation in correlated_pairs:
                # Skip if one is already filtered
                if feature1 in features_to_filter or feature2 in features_to_filter:
                    continue
                
                # Decide which feature to keep
                score1 = feature_scores[feature1]['score']
                score2 = feature_scores[feature2]['score']
                
                # Must include features cannot be filtered
                if feature1 in self.must_include:
                    to_filter = feature2
                elif feature2 in self.must_include:
                    to_filter = feature1
                # Filter the one with lower score
                elif score1 >= score2:
                    to_filter = feature2
                else:
                    to_filter = feature1
                
                # Record feature to filter
                features_to_filter.add(to_filter)
                feature_scores[to_filter]['filter_reason'] = f"与{feature1 if to_filter == feature2 else feature2}高度相关 ({correlation:.4f})"
            
            print(f"基于相关性分析过滤掉 {len(features_to_filter)} 个冗余特征")
        
        return feature_scores

    def select_features(self, feature_cols, train_df=None, importance_dict=None, 
                        psi_results=None, feature_stats=None, max_features=200):
        """
        Select features based on multiple metrics
        
        Args:
            feature_cols: Original feature list
            train_df: Training dataset for correlation analysis
            importance_dict: Feature importance dictionary
            psi_results: PSI stability results
            feature_stats: Feature statistics
            max_features: Maximum features to keep
            
        Returns:
            selected_features: Selected feature list
            feature_scores: Feature score details
        """
        start_time = time.time()
        print(f"开始特征选择，总特征数: {len(feature_cols)}")
        
        # Remove must exclude features
        feature_cols = [f for f in feature_cols if f not in self.must_exclude]
        
        # If no evaluation metrics and features <= limit, return directly
        if (importance_dict is None and psi_results is None and feature_stats is None and 
            len(feature_cols) <= max_features):
            print(f"无需筛选特征，直接返回所有 {len(feature_cols)} 个特征")
            return feature_cols, {}
        
        # Calculate feature scores
        feature_scores = self.calculate_feature_scores(
            feature_cols, importance_dict, psi_results, feature_stats
        )
        
        # Detect correlated features
        if train_df is not None:
            print("分析特征相关性以消除冗余...")
            # Sort top 200 features by score for correlation analysis
            top_features = [f for f, _ in sorted(
                feature_scores.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )][:200]
            feature_scores = self.detect_correlated_features(train_df, feature_scores, top_features)
        
        # Group features by must include/filter reason
        must_include_features = [f for f in feature_cols if f in self.must_include]
        filtered_features = {f: info for f, info in feature_scores.items() 
                           if info.get('filter_reason') is not None and f not in self.must_include}
        unfiltered_features = {f: info for f, info in feature_scores.items() 
                             if info.get('filter_reason') is None and f not in self.must_include}
        
        # Sort by score
        sorted_unfiltered = sorted(unfiltered_features.items(), key=lambda x: x[1]['score'], reverse=True)
        sorted_filtered = sorted(filtered_features.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Calculate needed feature count
        remaining_slots = max(0, max_features - len(must_include_features) - len(sorted_unfiltered))
        
        # Build final feature list
        final_features = must_include_features + [f for f, _ in sorted_unfiltered]
        if remaining_slots > 0 and sorted_filtered:
            additional_features = [f for f, _ in sorted_filtered[:remaining_slots]]
            final_features.extend(additional_features)
        
        # Ensure max_features limit
        features_to_keep = final_features[:max_features]
        
        # Output selection statistics
        print(f"\n特征筛选结果 (耗时: {time.time() - start_time:.2f}秒):")
        print(f"必须包含的特征: {len(must_include_features)}")
        print(f"未被过滤的特征: {len(unfiltered_features)}")
        print(f"被过滤的特征: {len(filtered_features)}")
        print(f"从被过滤特征中额外选择: {min(remaining_slots, len(sorted_filtered))}")
        print(f"最终保留的特征: {len(features_to_keep)}")
        
        # Summarize filter reasons
        if filtered_features:
            filter_reasons = {}
            for _, info in filtered_features.items():
                reason = info['filter_reason']
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            
            print("\n按过滤原因统计:")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} 个特征")
            
            # Output additionally selected filtered features
            if remaining_slots > 0 and sorted_filtered:
                print("\n被过滤但额外选择的特征:")
                for f, info in sorted_filtered[:remaining_slots]:
                    print(f"  {f}: {info['filter_reason']} [得分: {info['score']:.4f}]")
        
        return features_to_keep, feature_scores

def trim_features_by_importance(feature_cols: List[str], 
                               importance_dict: Optional[Dict[str, float]] = None, 
                               max_features: int = 200,
                               psi_results: Optional[Dict] = None, 
                               feature_stats: Optional[pd.DataFrame] = None, 
                               train_df: Optional[pd.DataFrame] = None, 
                               min_importance_pct: float = 0.01,
                               max_psi: float = 0.25, 
                               max_missing_rate: float = 0.99, 
                               min_variance: Optional[float] = None, 
                               min_iv: float = 0.02, 
                               correlation_threshold: float = 0.9, 
                               must_include: Optional[List[str]] = None, 
                               must_exclude: Optional[List[str]] = None, 
                               weights: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Trim feature list based on importance and quality metrics.
    
    Args:
        feature_cols: Feature column list
        importance_dict: Feature importance dictionary
        max_features: Maximum features to keep
        psi_results: PSI stability analysis results
        feature_stats: Feature statistics DataFrame
        train_df: Training data for correlation analysis
        min_importance_pct: Minimum importance percentage threshold
        max_psi: Maximum PSI threshold
        max_missing_rate: Maximum missing rate
        min_variance: Minimum variance threshold
        min_iv: Minimum IV value
        correlation_threshold: Feature correlation threshold
        must_include: Features that must be included
        must_exclude: Features that must be excluded
        weights: Weight dictionary for scoring factors
        
    Returns:
        List of features to keep
    """
    # Default optimized weights (focus on features correlated with positive samples)
    optimized_weights = {
        'importance': 0.4,    # Model importance
        'psi': 0.1,           # Stability weight reduced
        'missing': 0.1,       # Missing rate
        'variance': 0.1,      # Variance
        'iv': 0.3             # IV weight increased (focus on target correlation)
    }
    
    # Use optimized weights if none provided
    if weights is None:
        weights = optimized_weights
    
    print(f"开始特征选择，总特征数: {len(feature_cols)}")
    print(f"使用的权重: {weights}")
    
    # Create feature selector
    selector = FeatureSelector(
        weights=weights,
        min_importance_pct=min_importance_pct,
        max_psi=max_psi,
        max_missing_rate=max_missing_rate,
        min_variance=min_variance,
        min_iv=min_iv,
        correlation_threshold=correlation_threshold,
        must_include=must_include,
        must_exclude=must_exclude
    )
    
    # If training data exists, analyze feature correlation with positive samples
    if train_df is not None and 'label_apply' in train_df.columns:
        target_col = 'label_apply'
        print(f"分析特征与正样本({target_col}=1)的相关性...")
        
        # Create positive sample indicator
        pos_only = (train_df[target_col] == 1).astype(int)
        
        # Calculate feature correlation with positive samples
        pos_corr = {}
        for feature in tqdm(feature_cols, desc="计算正样本相关性"):
            if feature in train_df.columns and feature != target_col:
                pos_corr[feature] = train_df[feature].corr(pos_only)
        
        # Find features most correlated with positive samples
        top_pos_features = sorted(pos_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:100]
        
        print("\n与正样本最相关的前20个特征:")
        for feature, corr in top_pos_features[:20]:
            print(f"  {feature}: 相关系数 = {corr:.4f}")
        
        # Add these features to must-include list
        # Only select features with significant correlation (abs(corr) > 0.05)
        pos_feature_cutoff = 0.05
        pos_must_include = [f for f, c in top_pos_features if abs(c) > pos_feature_cutoff]
        
        print(f"\n基于正样本相关性，添加 {len(pos_must_include)} 个必选特征 (相关系数绝对值 > {pos_feature_cutoff})")
        
        # Merge must-include lists
        if must_include is None:
            must_include = pos_must_include
        else:
            must_include = list(set(must_include + pos_must_include))
        
        selector.must_include = set(must_include)
    
    # If feature statistics exist, find features most valuable for positive sample prediction
    if feature_stats is not None and not feature_stats.empty and 'iv' in feature_stats.columns:
        # Add high IV features to must-include list
        high_iv_threshold = feature_stats['iv'].quantile(0.8)  # Top 20% by IV
        high_iv_features = feature_stats[feature_stats['iv'] > high_iv_threshold]['feature'].tolist()
        
        print(f"\n基于信息值(IV>{high_iv_threshold:.4f})，添加 {len(high_iv_features)} 个高IV值特征")
        
        # Merge must-include lists
        if must_include is None:
            must_include = high_iv_features
        else:
            must_include = list(set(must_include + high_iv_features))
        
        selector.must_include = set(must_include)
    
    # Execute feature selection
    selected_features, feature_scores = selector.select_features(
        feature_cols=feature_cols,
        train_df=train_df,
        importance_dict=importance_dict,
        psi_results=psi_results,
        feature_stats=feature_stats,
        max_features=max_features
    )
    
    # Save feature scores and selection results
    model_dir = "funnel_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save feature selection results to CSV
    if feature_scores:
        scores_df = []
        for feature, scores in feature_scores.items():
            row = {'feature': feature, 'selected': feature in selected_features}
            for k, v in scores.items():
                if isinstance(v, (int, float, bool, str)):
                    row[k] = v
            scores_df.append(row)
        
        scores_df = pd.DataFrame(scores_df)
        scores_file = os.path.join(model_dir, "feature_selection_scores.csv")
        scores_df.to_csv(scores_file, index=False)
        print(f"特征选择详细评分已保存至 {scores_file}")
    
    return selected_features 