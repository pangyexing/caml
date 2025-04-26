#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature analysis and selection functionality.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.core.config import MODEL_DIR


def analyze_features_for_selection_parallel(
    train_df: pd.DataFrame, 
    feature_cols: List[str], 
    target: str = 'label_apply',
    n_jobs: int = 4
) -> pd.DataFrame:
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    feature_stats.to_csv(os.path.join(MODEL_DIR, "feature_selection_stats.csv"), index=False)
    
    # Provide feature selection suggestions
    print(f"\n特征分析完成，耗时 {time.time() - start_time:.2f} 秒")
    print("\n特征选择建议:")
    
    # High IV features
    high_iv_features = feature_stats.sort_values('iv', ascending=False).head(20)
    print("\n信息量(IV)最高的20个特征:")
    for idx, row in high_iv_features.iterrows():
        print(f"  {row['feature']}: IV={row['iv']:.4f}")
    
    # High correlation features
    high_corr_features = feature_stats.sort_values('correlation_with_target', ascending=False).head(20)
    print("\n与目标变量相关性最高的20个特征:")
    for idx, row in high_corr_features.iterrows():
        print(f"  {row['feature']}: 相关性={row['correlation_with_target']:.4f}")
    
    # High missing rate features
    high_missing_features = feature_stats[feature_stats['missing_or_zero_rate'] > 0.99]
    if not high_missing_features.empty:
        print("\n缺失或为0比例高于99%的特征 (建议考虑移除):")
        for idx, row in high_missing_features.iterrows():
            print(f"  {row['feature']}: 缺失或为0比例={row['missing_or_zero_rate']:.2%}")
    
    # Low variance features
    low_var_threshold = feature_stats['variance'].quantile(0.1)
    low_var_features = feature_stats[feature_stats['variance'] < low_var_threshold]
    if not low_var_features.empty:
        print("\n方差较低的特征 (建议考虑移除):")
        for idx, row in low_var_features.iterrows():
            print(f"  {row['feature']}: 方差={row['variance']:.6f}")
    
    return feature_stats


def analyze_positive_sample_subgroups(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    initial_features: List[str],
    target: str,
    initial_model,
    n_clusters: int = 3,
    sample_size: int = 10000
) -> List[str]:
    """
    Analyze positive sample subgroups to identify features important for different types of positive samples.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        initial_features: List of initial features
        target: Target variable
        initial_model: Trained XGBoost model
        n_clusters: Number of positive sample clusters to analyze
        sample_size: Maximum sample size for analysis
        
    Returns:
        List of important features for positive sample subgroups
    """
    print("\n=== 正样本子群体分析 ===")
    
    # Extract positive samples from test set
    pos_samples = test_df[test_df[target] == 1].copy()
    
    if len(pos_samples) < 10:
        print(f"正样本数量过少 ({len(pos_samples)}), 跳过子群体分析")
        return []
    
    # Limit sample size if needed
    if len(pos_samples) > sample_size:
        pos_samples = pos_samples.sample(sample_size, random_state=42)
    
    # Get features only
    X_pos = pos_samples[initial_features]
    
    # Fill missing values for clustering
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_pos_imputed = imputer.fit_transform(X_pos)
    
    # Cluster positive samples
    print(f"将正样本聚类为 {n_clusters} 个子群体...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pos_imputed)
    
    # Add cluster labels to the dataframe
    pos_samples['cluster'] = clusters
    
    # Calculate cluster sizes
    cluster_sizes = pos_samples['cluster'].value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        print(f"子群体 {cluster_id+1}: {size} 个样本 ({size/len(pos_samples):.1%})")
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(initial_model)
    
    # Find important features for each cluster
    important_features = []
    
    for cluster_id in range(n_clusters):
        cluster_samples = pos_samples[pos_samples['cluster'] == cluster_id]
        
        if len(cluster_samples) < 5:
            print(f"子群体 {cluster_id+1} 样本数量过少，跳过分析")
            continue
            
        print(f"\n分析子群体 {cluster_id+1} ({len(cluster_samples)} 个样本)...")
        
        try:
            # Get SHAP values for this cluster
            X_cluster = cluster_samples[initial_features]
            
            # 更全面的数据预处理
            X_cluster_processed = X_cluster.copy()
            
            # 1. 确保数据类型正确
            for col in X_cluster_processed.columns:
                if X_cluster_processed[col].dtype not in ['float64', 'int64']:
                    X_cluster_processed[col] = pd.to_numeric(X_cluster_processed[col], errors='coerce')
            
            # 2. 填充NaN值
            X_cluster_processed = X_cluster_processed.fillna(0)
            
            # 3. 处理无穷大值
            X_cluster_processed = X_cluster_processed.replace([np.inf, -np.inf], 0)
            
            # 4. 确保没有其他问题数据
            for col in X_cluster_processed.columns:
                if not np.isfinite(X_cluster_processed[col]).all():
                    X_cluster_processed[col] = X_cluster_processed[col].fillna(0)
            
            # 使用处理后的数据计算SHAP值
            # 设置较小的样本量以增加计算成功率
            sample_size = min(100, len(X_cluster_processed))
            if len(X_cluster_processed) > sample_size:
                X_sample = X_cluster_processed.sample(sample_size, random_state=42)
            else:
                X_sample = X_cluster_processed
                
            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample)
            
            # Average absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Map feature importance
            feature_importance = dict(zip(X_sample.columns, mean_shap))
            
            # Get most important features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"子群体 {cluster_id+1} 最重要特征:")
            for feature, importance in top_features:
                print(f"  - {feature}: {importance:.6f}")
                important_features.append(feature)
        
        except Exception as e:
            print(f"子群体 {cluster_id+1} 特征分析失败: {str(e)}")
            # 仍继续分析其他子群体
    
    # Remove duplicates while preserving order
    unique_important_features = []
    for f in important_features:
        if f not in unique_important_features:
            unique_important_features.append(f)
    
    print(f"\n子群体分析发现的重要特征数量: {len(unique_important_features)}")
    
    # Save important features list
    with open(os.path.join(MODEL_DIR, f"{target}_subgroup_features.txt"), 'w') as f:
        for feature in unique_important_features:
            f.write(f"{feature}\n")
    
    return unique_important_features


def analyze_feature_interactions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    initial_model,
    initial_features: List[str],
    top_n_features: int = 50,
    max_interactions: int = 25,
    sample_size: int = 2000,
    skip_shap_interactions: bool = False
) -> List[Tuple[str, str]]:
    """
    Analyze feature interactions to identify pairs with strong predictive power.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target: Target variable
        initial_model: Trained model
        initial_features: List of features
        top_n_features: Number of top features to consider for interactions
        max_interactions: Maximum number of interactions to return
        sample_size: Sample size for analysis
        skip_shap_interactions: If True, skip SHAP interaction calculation
        
    Returns:
        List of feature pairs (tuples) with strong interactions
    """
    
    print("\n=== 特征交互分析 ===")
    
    # Extract and sample positive instances
    pos_samples = test_df[test_df[target] == 1].copy()
    if len(pos_samples) < 10:
        print(f"正样本数量过少 ({len(pos_samples)}), 跳过特征交互分析")
        return []
    
    if len(pos_samples) > sample_size:
        pos_samples = pos_samples.sample(sample_size, random_state=42)
    
    # Get top important features
    feature_importance = initial_model.get_booster().get_score(importance_type='gain')
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
    top_feature_names = [f for f, _ in top_features]
    
    print(f"使用前 {len(top_feature_names)} 个重要特征进行交互分析...")
    
    # Approach 1: SHAP interaction values
    shap_interactions = get_shap_interactions(
        pos_samples, 
        initial_model, 
        skip_shap_interactions, 
        max_interactions
    )
    
    # Approach 2: Evaluate feature pairs with simple models
    model_interactions = get_model_interactions(
        test_df, 
        target, 
        top_feature_names, 
        max_interactions
    )
    
    # Merge interaction lists with priority to SHAP interactions
    all_interactions = []
    
    # Add SHAP interactions first
    all_interactions.extend(shap_interactions)
    
    # Add model interactions not already in the list
    for pair in model_interactions:
        if pair not in all_interactions and (pair[1], pair[0]) not in all_interactions:
            all_interactions.append(pair)
            if len(all_interactions) >= max_interactions:
                break
                
    # Limit to max interactions
    all_interactions = all_interactions[:max_interactions]
    
    print(f"\n识别出 {len(all_interactions)} 个重要的特征交互对")
    
    # Save interaction pairs
    interaction_file = os.path.join(MODEL_DIR, f"{target}_feature_interactions.txt")
    with open(interaction_file, 'w') as f:
        for feat1, feat2 in all_interactions:
            f.write(f"{feat1},{feat2}\n")
    
    return all_interactions


def get_shap_interactions(
    pos_samples: pd.DataFrame, 
    model,
    skip_shap: bool, 
    max_interactions: int,
    batch_size: int = 200
) -> List[Tuple[str, str]]:
    """
    Calculate SHAP interaction values for positive samples using batch processing.
    
    Args:
        pos_samples: DataFrame containing only positive samples.
        model: Trained XGBoost model.
        skip_shap: If True, skip SHAP calculation and return empty list.
        max_interactions: Maximum number of interaction pairs to return.
        batch_size: Number of samples to process in each SHAP batch.
        
    Returns:
        List of feature pairs (tuples) sorted by mean absolute SHAP interaction value.
    """
    import psutil
    
    if skip_shap:
        print("Skipping SHAP interaction calculation.")
        return []

    print("\n计算正样本 SHAP 交互值...")
    
    # Use features the model was trained on
    try:
        model_features = model.get_booster().feature_names
        if model_features is None:  # Fallback for older xgboost/sklearn versions
             model_features = model.feature_names_in_
        X_pos = pos_samples[model_features]
        n_features = len(model_features)
    except AttributeError:
         print("Error: Could not retrieve feature names from the model. Ensure the model is trained.")
         return []
    except KeyError as e:
        print(f"Error: Feature {e} not found in pos_samples DataFrame.")
        return []

    n_samples = len(X_pos)
    if n_samples == 0:
        print("No positive samples provided for SHAP interaction analysis.")
        return []

    print(f"分析 {n_samples} 个正样本，共 {n_features} 个特征")

    # Initialize accumulator for interaction values
    interaction_sum = np.zeros((n_features, n_features))
    total_processed = 0

    try:
        explainer = shap.TreeExplainer(model)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_pos.iloc[i:batch_end]
            
            # Memory check before calculating SHAP values
            mem = psutil.virtual_memory()
            print(f"处理批次 {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}... 内存使用: {mem.percent}%, 可用: {mem.available/1024/1024/1024:.2f}GB")
            
            if mem.available < 1 * 1024 * 1024 * 1024:  # Less than 1GB available
                 print("警告: SHAP 计算前检测到内存不足")

            try:
                # Calculate SHAP interaction values for this batch
                shap_interaction_values_batch = explainer.shap_interaction_values(X_batch)
                
                # Ensure the output shape is correct
                if isinstance(shap_interaction_values_batch, list):
                    # For multi-output models (e.g., multi-class), we take the first output
                    shap_interaction_values_batch = shap_interaction_values_batch[0]
                
                # Sum up interaction values (absolute)
                interaction_sum += np.abs(shap_interaction_values_batch).sum(axis=0)
                total_processed += len(X_batch)
                
            except Exception as e:
                print(f"批次 SHAP 计算错误: {str(e)[:100]}...")
                if i > 0:  # If we've processed some batches, we can still continue
                    print("继续使用已处理的批次...")
                    break
                else:
                    return []  # If the first batch fails, we return empty
                
        # Check if we processed any samples
        if total_processed == 0:
            print("无法处理任何样本，跳过 SHAP 交互分析")
            return []
            
        # Normalize by the number of samples processed
        interaction_sum /= total_processed
        
        # Create feature pair scores
        interaction_scores = []
        feature_names = model_features
        
        # Get the upper triangle of the interaction matrix (excluding diagonal)
        for i in range(n_features):
            for j in range(i+1, n_features):
                score = interaction_sum[i, j]
                if not np.isnan(score):
                    interaction_scores.append((feature_names[i], feature_names[j], score))
        
        # Sort by interaction strength
        interaction_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Extract top pairs
        top_interactions = [(pair[0], pair[1]) for pair in interaction_scores[:max_interactions]]
        
        print(f"SHAP 交互分析完成，找到 {len(top_interactions)} 个重要交互对")
        return top_interactions
        
    except Exception as e:
        print(f"SHAP 交互分析错误: {str(e)}")
        return []


def get_model_interactions(
    test_df: pd.DataFrame,
    target: str,
    top_feature_names: List[str],
    max_interactions: int
) -> List[Tuple[str, str]]:
    """
    Evaluate feature pairs with simple models to find interactions.
    
    Args:
        test_df: Testing DataFrame
        target: Target variable name
        top_feature_names: List of top feature names to consider
        max_interactions: Maximum number of interaction pairs to return
        
    Returns:
        List of feature pairs (tuples) with strong interactions
    """
    from itertools import combinations

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    print("\n使用简单模型评估特征交互...")
    
    if len(top_feature_names) < 2:
        print("特征数量不足，跳过模型交互分析")
        return []
    
    # Limit the number of features to consider to avoid combinatorial explosion
    max_features_to_consider = min(30, len(top_feature_names))
    features_to_consider = top_feature_names[:max_features_to_consider]
    
    # Generate all pairs
    feature_pairs = list(combinations(features_to_consider, 2))
    print(f"评估 {len(feature_pairs)} 个特征对...")
    
    # Function to evaluate a pair
    def evaluate_pair(pair):
        feat1, feat2 = pair
        
        try:
            # Extract features
            X = test_df[[feat1, feat2]].copy()
            y = test_df[target]
            
            # Fill NAs
            X = X.fillna(X.mean())
            
            # Train a simple model
            clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            clf.fit(X, y)
            
            # Predict
            y_pred = clf.predict_proba(X)[:, 1]
            
            # Calculate AUC
            auc = roc_auc_score(y, y_pred)
            
            # Create interaction feature and evaluate
            X['interaction'] = X[feat1] * X[feat2]
            
            # Train model with interaction
            clf_with_interaction = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            clf_with_interaction.fit(X, y)
            
            # Predict with interaction
            y_pred_with_interaction = clf_with_interaction.predict_proba(X)[:, 1]
            
            # Calculate AUC with interaction
            auc_with_interaction = roc_auc_score(y, y_pred_with_interaction)
            
            # Calculate improvement
            improvement = auc_with_interaction - auc
            
            return (feat1, feat2, improvement)
        except Exception as e:
            print(f"评估特征对 {feat1}, {feat2} 时出错: {str(e)[:100]}")
            return (feat1, feat2, -1.0)
    
    # Evaluate all pairs
    pair_scores = []
    for pair in tqdm(feature_pairs, desc="评估特征交互"):
        score = evaluate_pair(pair)
        if score[2] > 0:  # Only keep pairs with positive improvement
            pair_scores.append(score)
    
    # Sort by improvement
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Get top pairs
    top_pairs = [(pair[0], pair[1]) for pair in pair_scores[:max_interactions]]
    
    print(f"模型交互分析完成，找到 {len(top_pairs)} 个有效交互对")
    return top_pairs 