#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature selection functionality.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.config import FEATURE_SELECTION_PARAMS, MODEL_DIR


class FeatureSelector:
    """Feature selection and evaluation class."""
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None, 
        min_importance_pct: float = 0.01, 
        max_psi: float = 0.25, 
        max_missing_rate: float = 0.99, 
        min_variance: Optional[float] = None, 
        min_iv: float = 0.02,
        correlation_threshold: float = 0.9, 
        must_include: Optional[List[str]] = None, 
        must_exclude: Optional[List[str]] = None
    ):
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
    
    def _normalize_score(
        self, 
        value: float, 
        min_val: float, 
        max_val: float, 
        higher_is_better: bool = True
    ) -> float:
        """
        Normalize value to 0-1 range
        
        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range
            higher_is_better: Whether higher values are better
            
        Returns:
            Normalized value in 0-1 range
        """
        if min_val == max_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return normalized if higher_is_better else (1 - normalized)
    
    def score_importance(
        self, 
        feature: str, 
        importance_dict: Dict[str, float], 
        total_importance: float
    ) -> Tuple[float, Optional[str]]:
        """
        Score feature importance
        
        Args:
            feature: Feature name
            importance_dict: Dictionary of feature importances
            total_importance: Sum of all importances
            
        Returns:
            Tuple of (importance score, filter reason)
        """
        if feature not in importance_dict:
            return 0, None
        
        importance = importance_dict[feature]
        rel_importance = importance / total_importance
        
        # Flag features below threshold
        filter_reason = f"低重要性 ({rel_importance:.4f})" if rel_importance < self.min_importance_pct else None
        
        return rel_importance, filter_reason
    
    def score_psi(
        self, 
        feature: str, 
        psi_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, Optional[str]]:
        """
        Score feature stability (PSI)
        
        Args:
            feature: Feature name
            psi_results: PSI results dictionary
            
        Returns:
            Tuple of (PSI score, filter reason)
        """
        if feature not in psi_results:
            return 0.5, None  # Default medium stability
        
        psi_avg = psi_results[feature]['avg_psi']
        psi_score = 1.0 - min(psi_avg / self.max_psi, 1.0)
        
        # Flag unstable features
        filter_reason = f"不稳定 (PSI={psi_avg:.4f})" if psi_avg > self.max_psi else None
        
        return psi_score, filter_reason
    
    def score_stats(
        self, 
        feature: str, 
        stats_dict: Dict[str, pd.Series]
    ) -> Tuple[float, Optional[str]]:
        """
        Score based on feature statistics (missing rate, variance, IV)
        
        Args:
            feature: Feature name
            stats_dict: Dictionary of feature statistics
            
        Returns:
            Tuple of (combined stats score, filter reason)
        """
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
    
    def calculate_feature_scores(
        self, 
        feature_cols: List[str], 
        importance_dict: Optional[Dict[str, float]] = None, 
        psi_results: Optional[Dict[str, Dict[str, Any]]] = None, 
        feature_stats: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, Any]]:
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
    
    def detect_correlated_features(
        self, 
        train_df: pd.DataFrame, 
        feature_scores: Dict[str, Dict[str, Any]], 
        top_features: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Detect and handle highly correlated features
        
        Args:
            train_df: Training DataFrame
            feature_scores: Feature scores dictionary
            top_features: Optional list of top features to check
            
        Returns:
            Dictionary of correlated feature groups
        """
        import time
        
        # If no top features provided, use all features with scores
        if top_features is None:
            top_features = list(feature_scores.keys())
        
        # Calculate correlation matrix for these features
        print(f"计算特征相关性矩阵，特征数量 = {len(top_features)}...")
        start_time = time.time()
        
        # Select features that exist in the dataframe
        valid_features = [f for f in top_features if f in train_df.columns]
        
        if len(valid_features) < 2:
            print("特征数量不足，无法计算相关性")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = train_df[valid_features].corr().abs()
        
        # Find features with correlations above threshold
        correlated_pairs = []
        
        # Get pairs with high correlation (excluding self-correlations)
        for i in range(len(valid_features)):
            for j in range(i+1, len(valid_features)):
                feat1 = valid_features[i]
                feat2 = valid_features[j]
                
                corr = corr_matrix.loc[feat1, feat2]
                
                if corr >= self.correlation_threshold:
                    correlated_pairs.append((feat1, feat2, corr))
        
        # Sort by correlation
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Group correlated features
        correlated_groups = {}
        
        for feat1, feat2, corr in correlated_pairs:
            # Check if either feature already belongs to a group
            feat1_group = None
            feat2_group = None
            
            for group_id, group in correlated_groups.items():
                if feat1 in [p[0] for p in group]:
                    feat1_group = group_id
                if feat2 in [p[0] for p in group]:
                    feat2_group = group_id
            
            if feat1_group is None and feat2_group is None:
                # Create a new group
                group_id = len(correlated_groups) + 1
                correlated_groups[group_id] = [(feat1, 0), (feat2, corr)]
            elif feat1_group is not None and feat2_group is None:
                # Add feat2 to feat1's group
                correlated_groups[feat1_group].append((feat2, corr))
            elif feat1_group is None and feat2_group is not None:
                # Add feat1 to feat2's group
                correlated_groups[feat2_group].append((feat1, corr))
            elif feat1_group != feat2_group:
                # Merge the two groups
                merged_group = correlated_groups[feat1_group] + correlated_groups[feat2_group]
                correlated_groups[feat1_group] = merged_group
                del correlated_groups[feat2_group]
        
        print(f"相关性分析完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"找到 {len(correlated_pairs)} 个高相关特征对，分为 {len(correlated_groups)} 个组")
        
        # Print correlation groups
        for group_id, group in correlated_groups.items():
            print(f"\n相关特征组 {group_id}:")
            for feat, corr in group:
                # Find the feature's score
                score = feature_scores.get(feat, {}).get('score', 0)
                print(f"  {feat}: 相关性 = {corr:.4f}, 特征分数 = {score:.4f}")
        
        return correlated_groups
    
    def select_features(
        self, 
        feature_cols: List[str], 
        train_df: Optional[pd.DataFrame] = None, 
        importance_dict: Optional[Dict[str, float]] = None, 
        psi_results: Optional[Dict] = None, 
        feature_stats: Optional[pd.DataFrame] = None, 
        max_features: int = FEATURE_SELECTION_PARAMS['max_features']
    ) -> List[str]:
        """
        Select features based on multiple criteria.
        
        Args:
            feature_cols: List of feature columns to consider
            train_df: Training DataFrame
            importance_dict: Dictionary of feature importances
            psi_results: PSI stability results
            feature_stats: Feature statistics DataFrame
            max_features: Maximum number of features to select
            
        Returns:
            List of selected features
        """
        print(f"\n开始特征选择，候选特征数量: {len(feature_cols)}")
        
        # Calculate feature scores
        self.feature_scores = self.calculate_feature_scores(
            feature_cols, 
            importance_dict, 
            psi_results, 
            feature_stats
        )
        
        # Sort features by score
        sorted_features = sorted(
            self.feature_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # Handle correlated features if train_df is provided
        if train_df is not None:
            # Get top features (more than max_features)
            top_feature_limit = min(max_features * 2, len(sorted_features))
            top_features = [f[0] for f in sorted_features[:top_feature_limit]]
            
            # Detect correlated features
            correlated_groups = self.detect_correlated_features(train_df, self.feature_scores, top_features)
            
            # Handle correlated features by keeping the one with highest score
            for group_id, group in correlated_groups.items():
                # Sort group by score
                sorted_group = sorted(
                    [(feat, corr, self.feature_scores.get(feat, {}).get('score', 0)) for feat, corr in group], 
                    key=lambda x: x[2], 
                    reverse=True
                )
                
                # Keep the first feature, mark others as duplicates
                kept_feature = sorted_group[0][0]
                for feat, corr, score in sorted_group[1:]:
                    if feat in self.feature_scores:
                        self.feature_scores[feat]['filter_reason'] = f"与特征 {kept_feature} 高相关 (r={corr:.4f})"
                        self.feature_scores[feat]['score'] *= 0.5  # Reduce score of correlated features
        
        # Get final filtered list (remove filtered features)
        selected_features = []
        filtered_features = []
        
        # Always include must-include features
        for feature in self.must_include:
            if feature in feature_cols:
                selected_features.append(feature)
                
        # Add other top features up to max_features limit
        features_to_add = max_features - len(selected_features)
        
        for feature, score_info in sorted_features:
            # Skip if already in selected features (must-include)
            if feature in selected_features:
                continue
                
            # Skip if has filter reason
            if score_info.get('filter_reason'):
                filtered_features.append((feature, score_info.get('filter_reason')))
                continue
                
            # Add feature if under limit
            if len(selected_features) < max_features:
                selected_features.append(feature)
            else:
                break
                
        # Print selection results
        print(f"\n特征选择完成，选择了 {len(selected_features)} 个特征 (最大限制: {max_features})")
        print(f"  - 必选特征数量: {len(self.must_include)}")
        print(f"  - 根据筛选规则排除的特征数量: {len(filtered_features)}")
        
        if filtered_features:
            print("\n排除特征示例:")
            for feature, reason in filtered_features[:10]:  # Show only top 10
                print(f"  - {feature}: {reason}")
            
            if len(filtered_features) > 10:
                print(f"  ... 及其他 {len(filtered_features) - 10} 个特征")
                
        # Export filtered features
        filtered_df = pd.DataFrame([
            {'feature': f, 'reason': r} for f, r in filtered_features
        ])
        
        if not filtered_df.empty:
            filtered_df.to_csv(os.path.join(MODEL_DIR, "filtered_features.csv"), index=False)
        
        return selected_features


def trim_features_by_importance(
    feature_cols: List[str], 
    importance_dict: Optional[Dict[str, float]] = None, 
    max_features: int = FEATURE_SELECTION_PARAMS['max_features'],
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
    weights: Optional[Dict[str, float]] = None,
    return_scores: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, Dict[str, Any]]]]:
    """
    Convenience function to trim features using the FeatureSelector class.
    
    Args:
        feature_cols: List of feature columns
        importance_dict: Feature importance dictionary
        max_features: Maximum number of features to select
        psi_results: PSI stability results
        feature_stats: Feature statistics DataFrame
        train_df: Training DataFrame for correlation analysis
        min_importance_pct: Minimum importance percentage
        max_psi: Maximum PSI threshold
        max_missing_rate: Maximum missing rate
        min_variance: Minimum variance threshold
        min_iv: Minimum IV value threshold
        correlation_threshold: Feature correlation threshold
        must_include: Features that must be included
        must_exclude: Features that must be excluded
        weights: Weight dictionary for scoring factors
        return_scores: Whether to return feature scores along with selected features
        
    Returns:
        If return_scores is False:
            List of selected features
        If return_scores is True:
            Tuple of (List of selected features, Dictionary of feature scores)
    """
    # Initialize feature selector
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
    
    # Select features
    selected_features = selector.select_features(
        feature_cols=feature_cols,
        train_df=train_df,
        importance_dict=importance_dict,
        psi_results=psi_results,
        feature_stats=feature_stats,
        max_features=max_features
    )
    
    # Save selected features to file
    with open(os.path.join(MODEL_DIR, "selected_features.txt"), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
            
    # Return results based on return_scores parameter
    if return_scores:
        # Get feature scores for visualization
        feature_scores = selector.calculate_feature_scores(
            feature_cols=feature_cols,
            importance_dict=importance_dict,
            psi_results=psi_results,
            feature_stats=feature_stats
        )
        
        # Add selection parameters to the scores dictionary for reference
        feature_scores['_params'] = {
            'min_importance_pct': min_importance_pct,
            'max_psi': max_psi,
            'max_missing_rate': max_missing_rate,
            'min_variance': min_variance,
            'min_iv': min_iv,
            'correlation_threshold': correlation_threshold,
            'weights': weights or selector.default_weights
        }
        
        # 增强feature_scores字典，确保它包含所有必要的特征统计信息
        # 创建feature_stats的便捷查询字典
        stats_dict = {}
        if feature_stats is not None and not feature_stats.empty:
            for _, row in feature_stats.iterrows():
                if 'feature' in row:
                    stats_dict[row['feature']] = row
        
        # 遍历所有特征，确保每个特征都有完整的统计信息
        for feature, scores in feature_scores.items():
            if feature == '_params':  # 跳过参数字典
                continue
                
            # 确保必要的键存在
            for key in ['importance', 'psi', 'missing_rate', 'variance', 'iv', 'correlation_with_target']:
                if key not in scores:
                    scores[key] = 0.0  # 设置默认值
            
            # 从feature_stats添加详细统计数据
            if feature in stats_dict:
                stats_row = stats_dict[feature]
                scores['missing_rate'] = stats_row.get('missing_or_zero_rate', 0.0)
                scores['variance'] = stats_row.get('variance', 0.0)
                scores['iv'] = stats_row.get('iv', 0.0)
                scores['correlation_with_target'] = abs(stats_row.get('correlation_with_target', 0.0))
                
            # 从importance_dict添加重要性分数
            if importance_dict and feature in importance_dict:
                total_importance = sum(importance_dict.values()) if importance_dict else 1.0
                scores['importance'] = importance_dict[feature] / total_importance if total_importance > 0 else 0.0
                
            # 从psi_results添加稳定性分数
            if psi_results and feature in psi_results:
                psi_value = psi_results[feature].get('avg_psi', 0.0)
                scores['psi'] = 1.0 - min(psi_value / max_psi, 1.0) if max_psi > 0 else 0.0
        
        # 打印数据检查信息
        print(f"\n特征评分数据增强完成，特征数量: {len(feature_scores) - 1}")  # -1 因为包含_params
        
        return selected_features, feature_scores
    
    return selected_features 