#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core modeling module.
Handles model training, evaluation, and pipeline workflows.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    recall_score, 
    precision_score,
    f1_score,
    roc_curve
)
import shap
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

# Import local modules
from src.preprocessing import preprocess_data
from src.feature_engineering import (
    analyze_feature_stability,
    analyze_features_for_selection_parallel,
    trim_features_by_importance
)
# Import shared evaluation and model modules
from src.evaluation import (
    create_score_bins,
    plot_score_distribution,
    evaluate_predictions,
    plot_precision_recall_curve
)
from src.model_utils import (
    load_pmml_model,
    get_model_features,
    prepare_prediction_data,
    predict_with_model,
    save_feature_list,
    load_feature_list
)
# Import configuration module
from src.config import FEATURE_SELECTION_PARAMS, FEATURE_SELECTION_WEIGHTS, MODEL_DIR
# Import deployment functions
from src.deployment import deploy_model

def train_funnel_model(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      target: str = 'label_apply') -> Tuple[PMMLPipeline, XGBClassifier, Dict[str, Any], Dict[str, float]]:
    """
    Train a funnel conversion model optimized for positive sample prediction.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target: Target label (default: 'label_apply')
    
    Returns:
        model: Trained XGBoost model
        metrics: Evaluation metrics
        feature_importance: Feature importance dictionary
    """
    exclude_cols = ['input_key', 'recall_date', 'label_register', 'label_apply', 'label_approve', 'time_bin', 'score']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols and col in train_df.columns]

    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Calculate class weights for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"训练集：正样本数量={pos_count}，负样本数量={neg_count}，正负样本比例=1:{neg_count/pos_count:.2f}")
    
    # Optimized parameters for positive sample prediction
    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,                   # Reduced tree depth to prevent overfitting
        'learning_rate': 0.05,            # Lower learning rate to better capture positive sample features
        'subsample': 0.85,                # Increased subsample ratio
        'colsample_bytree': 0.8,          # Feature column sampling ratio
        'min_child_weight': 2,            # Reduced minimum child weight to help capture minority class
        'scale_pos_weight': pos_weight,   # Increase positive sample weight
        'gamma': 0.1,                     # Add regularization to control overfitting
        'reg_alpha': 0.01,                # L1 regularization
        'reg_lambda': 1.0,                # L2 regularization
        'use_label_encoder': False,
        'seed': 42
    }

    # Use PR-AUC as early stopping metric, focusing on positive sample prediction
    xgb_clf = XGBClassifier(**params)
    pipeline = PMMLPipeline([("classifier", xgb_clf)])

    # Set evaluation metric to PR-AUC
    eval_set = [(X_test, y_test)]
    pipeline.fit(X_train, y_train,
                classifier__eval_set=eval_set,
                classifier__eval_metric='aucpr',   # Use PR-AUC instead of AUC
                classifier__early_stopping_rounds=20,
                classifier__verbose=10)

    model = pipeline.named_steps['classifier']

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate F1 and F2 scores (F2 gives higher weight to recall)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    f2_scores = 5 * precision * recall / (4 * precision + recall + 1e-10)
    
    # Find threshold with maximum F2 score to improve recall
    best_f2_idx = np.argmax(f2_scores[:-1]) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_f2_idx] if len(thresholds) > 0 else 0.5
    
    # Also keep F1 max threshold as alternative
    best_f1_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    best_f1_threshold = thresholds[best_f1_idx] if len(thresholds) > 0 else 0.5

    # Test different thresholds
    threshold_results = []
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, best_f1_threshold, best_threshold]
    for threshold in test_thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        threshold_results.append((threshold, prec, rec, f1))
    
    print("\n测试不同阈值的效果:")
    for t, p, r, f in threshold_results:
        print(f"阈值: {t:.3f}, 精确率: {p:.4f}, 召回率: {r:.4f}, F1: {f:.4f}")

    # Evaluate with optimized threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ks_score = max(tpr - fpr)

    # Collect metrics
    f1_at_best_threshold = f1_score(y_test, y_pred)
    recall_at_best_threshold = recall_score(y_test, y_pred)
    precision_at_best_threshold = precision_score(y_test, y_pred)
    
    metrics = {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'ks': ks_score,
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
        'precision': precision_at_best_threshold,
        'recall': recall_at_best_threshold,
        'f1': f1_at_best_threshold,
        'f1_threshold': best_f1_threshold,
        'threshold_results': threshold_results
    }
    
    print(f"\n优化指标 (阈值={best_threshold:.4f}):")
    print(f"AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}, KS: {ks_score:.4f}")
    print(f"精确率: {precision_at_best_threshold:.4f}, 召回率: {recall_at_best_threshold:.4f}, F1: {f1_at_best_threshold:.4f}")
    
    # Get feature importance and visualization
    feature_importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    
    top_n = 20
    plot_features = [item[0] for item in sorted_importance[:top_n]]
    plot_scores = [item[1] for item in sorted_importance[:top_n]]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(plot_scores)), plot_scores[::-1], align='center')
    plt.yticks(range(len(plot_scores)), plot_features[::-1])
    plt.xlabel('F score (Gain)')
    plt.ylabel('Features')
    plt.title(f'Feature Importance (Gain) for {target} (Top {top_n})')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target}.png')
    plt.close()
    
    # Export feature importance
    os.makedirs(MODEL_DIR, exist_ok=True)
    importance_file = os.path.join(MODEL_DIR, f"{target}_feature_importance.csv")
    
    importance_df = pd.DataFrame(sorted_importance, columns=['feature', 'importance'])
    importance_df.to_csv(importance_file, index=False)
    
    # SHAP analysis for more stable feature importance evaluation
    sample_size = min(1000, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance for {target}')
    plt.tight_layout()
    plt.savefig(f'shap_importance_{target}.png')
    plt.close()

    plt.figure(figsize=(14, 10)) 
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'SHAP Summary Plot for {target}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{target}.png')
    plt.close() 
        
    # Calculate average absolute SHAP value per feature
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_dict = dict(zip(X_sample.columns, shap_importance))
    
    # Sort and export
    shap_importance_file = os.path.join(MODEL_DIR, f"{target}_shap_importance.csv")
    
    shap_items = sorted(shap_importance_dict.items(), key=lambda item: item[1], reverse=True)
    shap_df = pd.DataFrame(shap_items, columns=['feature', 'importance'])
    shap_df.to_csv(shap_importance_file, index=False)

    # Save model as PMML file
    model_path = os.path.join(MODEL_DIR, f"{target}_model.pmml")
    sklearn2pmml(pipeline, model_path, with_repr=True)
    # Return the fitted pipeline so downstream exports preserve column names
    return pipeline, model, metrics, feature_importance

def analyze_positive_sample_subgroups(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    initial_features: List[str],
    target: str,
    initial_model: XGBClassifier,
    n_clusters: int = 3,
    sample_size: int = 1000
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
    from sklearn.cluster import KMeans
    import shap
    
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
        
        # Get SHAP values for this cluster
        X_cluster = cluster_samples[initial_features]
        shap_values = explainer.shap_values(X_cluster)
        
        # Average absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_dict = dict(zip(initial_features, mean_shap))
        
        # Get top features for this cluster
        top_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print(f"子群体 {cluster_id+1} 重要特征:")
        for f, v in top_features[:5]:
            print(f"  {f}: {v:.6f}")
        
        # Add to important features list
        important_features.extend([f for f, _ in top_features])
    
    # Remove duplicates while preserving order
    unique_important_features = []
    for f in important_features:
        if f not in unique_important_features:
            unique_important_features.append(f)
    
    print(f"\n子群体分析发现的重要特征数量: {len(unique_important_features)}")
    return unique_important_features

def analyze_feature_interactions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    initial_model: XGBClassifier,
    initial_features: List[str],
    top_n_features: int = 30,
    max_interactions: int = 15,
    sample_size: int = 1000
) -> List[Tuple[str, str]]:
    """
    Analyze feature interactions to identify pairs with strong predictive power for positive samples.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target: Target variable
        initial_model: Trained model
        initial_features: List of features
        top_n_features: Number of top features to consider for interactions
        max_interactions: Maximum number of interactions to return
        sample_size: Sample size for analysis
        
    Returns:
        List of feature pairs (tuples) with strong interactions
    """
    import shap
    from itertools import combinations
    from sklearn.metrics import roc_auc_score
    
    print("\n=== 特征交互分析 ===")
    
    # Extract positive samples from test set
    pos_samples = test_df[test_df[target] == 1].copy()
    
    if len(pos_samples) < 10:
        print(f"正样本数量过少 ({len(pos_samples)}), 跳过特征交互分析")
        return []
    
    # Limit sample size if needed
    if len(pos_samples) > sample_size:
        pos_samples = pos_samples.sample(sample_size, random_state=42)
    
    # Get features only
    X_pos = pos_samples[initial_features]
    
    # Get top important features according to model
    feature_importance = initial_model.get_booster().get_score(importance_type='gain')
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
    top_feature_names = [f for f, _ in top_features]
    
    print(f"使用前 {len(top_feature_names)} 个重要特征进行交互分析...")
    
    # Approach 1: Use SHAP interaction values to find important feature interactions
    try:
        print("计算SHAP交互值...")
        # Use a smaller sample for interaction analysis as it's computationally expensive
        interaction_sample_size = min(500, len(pos_samples))
        interaction_sample = pos_samples.sample(interaction_sample_size, random_state=42)
        X_interaction = interaction_sample[top_feature_names]
        
        # Initialize SHAP explainer and compute interaction values
        explainer = shap.TreeExplainer(initial_model)
        interaction_values = explainer.shap_interaction_values(X_interaction)
        
        # Sum absolute interaction values across samples
        interaction_sum = np.abs(interaction_values).sum(axis=0)
        
        # Extract top interactions (excluding self-interactions)
        interaction_scores = []
        for i, feat_i in enumerate(top_feature_names):
            for j, feat_j in enumerate(top_feature_names):
                if i < j:  # Only include each pair once
                    interaction_scores.append(
                        (feat_i, feat_j, interaction_sum[i, j] + interaction_sum[j, i])
                    )
        
        # Sort interactions by score
        interaction_scores.sort(key=lambda x: x[2], reverse=True)
        
        shap_interactions = [(f1, f2) for f1, f2, _ in interaction_scores[:max_interactions]]
        
        print("\n基于SHAP交互值的重要特征对:")
        for i, (f1, f2, score) in enumerate(interaction_scores[:10]):
            print(f"  {i+1}. {f1} × {f2}: {score:.6f}")
            
    except Exception as e:
        print(f"SHAP交互值分析出错: {e}")
        shap_interactions = []
    
    # Approach 2: Evaluate feature pairs using simple models
    print("\n评估特征对的预测能力...")
    
    # Get a balanced sample for pair evaluation
    pos_eval = test_df[test_df[target] == 1].sample(
        min(200, test_df[target].sum()), 
        random_state=42
    )
    neg_eval = test_df[test_df[target] == 0].sample(
        min(200, len(test_df) - test_df[target].sum()), 
        random_state=42
    )
    eval_sample = pd.concat([pos_eval, neg_eval])
    y_eval = eval_sample[target]
    
    # Function to evaluate feature pair
    def evaluate_feature_pair(feat1, feat2):
        try:
            # Create product interaction
            X_pair = eval_sample[[feat1, feat2]].copy()
            
            # Fill missing values with zeros
            X_pair.fillna(0, inplace=True)
            
            # Create interaction feature
            X_pair['interaction'] = X_pair[feat1] * X_pair[feat2]
            
            # Train a simple model using just these features
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            clf.fit(X_pair, y_eval)
            
            # Predict and evaluate
            y_pred = clf.predict_proba(X_pair)[:, 1]
            auc_score = roc_auc_score(y_eval, y_pred)
            
            # Get feature importance to see if interaction is useful
            importances = clf.feature_importances_
            interaction_importance = importances[2]  # importance of interaction feature
            
            return auc_score, interaction_importance
        except Exception as e:
            print(f"评估特征对 {feat1} × {feat2} 出错: {e}")
            return 0.5, 0.0
    
    # Evaluate all pairs from top features
    feature_pairs = list(combinations(top_feature_names[:15], 2))
    pair_scores = []
    
    for feat1, feat2 in feature_pairs:
        auc_score, interaction_importance = evaluate_feature_pair(feat1, feat2)
        # Only consider pairs where interaction has meaningful contribution
        if interaction_importance > 0.1:
            pair_scores.append((feat1, feat2, auc_score, interaction_importance))
    
    # Sort by AUC score
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    
    print("\n基于模型评估的重要特征对:")
    for i, (f1, f2, auc, imp) in enumerate(pair_scores[:10]):
        print(f"  {i+1}. {f1} × {f2}: AUC={auc:.4f}, 交互重要性={imp:.4f}")
    
    # Combine results from both approaches
    model_interactions = [(f1, f2) for f1, f2, _, _ in pair_scores[:max_interactions]]
    
    # Merge interaction lists with priority to SHAP interactions
    all_interactions = []
    all_interactions.extend(shap_interactions)
    
    # Add model interactions not already in the list
    for pair in model_interactions:
        if pair not in all_interactions and (pair[1], pair[0]) not in all_interactions:
            all_interactions.append(pair)
    
    # Limit to max interactions
    all_interactions = all_interactions[:max_interactions]
    
    print(f"\n识别出 {len(all_interactions)} 个重要的特征交互对")
    return all_interactions

def two_stage_modeling_pipeline(train_df: pd.DataFrame, 
                               test_df: pd.DataFrame, 
                               target: str = 'label_apply',
                               resume_from: str = None,
                               checkpoint_dir: str = None) -> Dict[str, Any]:
    """
    Two-stage modeling pipeline: first train an initial model for feature selection,
    then retrain with selected features and deploy.
    Optimization goal: improve positive sample prediction and PR-AUC metric.
    
    Args:
        train_df: Training dataset
        test_df: Testing dataset
        target: Target variable
        resume_from: Stage to resume from (None means start from beginning)
        checkpoint_dir: Directory to save checkpoints (default is MODEL_DIR/checkpoints)
    
    Returns:
        Dictionary with final model, features, and evaluation metrics
    """
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints', target)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define checkpoint file paths
    checkpoint_status_file = os.path.join(checkpoint_dir, 'pipeline_status.json')
    initial_model_file = os.path.join(checkpoint_dir, 'initial_model.pkl')
    preprocess_checkpoint = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
    feature_analysis_file = os.path.join(checkpoint_dir, 'feature_analysis.pkl')
    selected_features_file = os.path.join(checkpoint_dir, 'selected_features.json')
    
    # Initialize or load checkpoint status
    if resume_from is None and os.path.exists(checkpoint_status_file):
        try:
            with open(checkpoint_status_file, 'r') as f:
                checkpoint_status = json.load(f)
                current_stage = checkpoint_status.get('current_stage', 'start')
                print(f"Found checkpoint status: {current_stage}")
        except Exception as e:
            print(f"Error loading checkpoint status: {e}")
            current_stage = 'start'
    elif resume_from:
        current_stage = resume_from
        print(f"Resuming from stage: {current_stage}")
    else:
        current_stage = 'start'
        
    # Update checkpoint status
    def update_checkpoint_status(stage):
        try:
            checkpoint_status = {
                'current_stage': stage,
                'last_updated': datetime.now().isoformat(),
                'target': target
            }
            with open(checkpoint_status_file, 'w') as f:
                json.dump(checkpoint_status, f)
            print(f"Updated checkpoint status to: {stage}")
        except Exception as e:
            print(f"Warning: Failed to update checkpoint status: {e}")
    
    # Initialize result dictionary
    result = {}
    
    # First stage: train initial model with all features
    if current_stage in ['start', 'preprocess']:
        print("=== 第一阶段：数据预处理 ===")
        update_checkpoint_status('preprocess')
        
        exclude_cols = ['input_key', 'recall_date', 'label_register', 'label_apply', 'label_approve', 'time_bin', 'score']
        initial_features = [col for col in train_df.columns if col not in exclude_cols]
        print(f"初始特征数量: {len(initial_features)}")
        
        # Analyze sample distribution
        pos_count = train_df[target].sum()
        total_count = len(train_df)
        pos_ratio = pos_count / total_count
        print(f"训练集正样本比例: {pos_count}/{total_count} = {pos_ratio:.2%}")
        
        # Preprocess training data
        try:
            print("预处理训练集和测试集...")
            train_df_processed = preprocess_data(train_df.copy(), initial_features)
            test_df_processed = preprocess_data(test_df.copy(), initial_features, is_training=False)
            
            # Save preprocessed data for checkpoint
            with open(preprocess_checkpoint, 'wb') as f:
                pickle.dump({
                    'train_df': train_df_processed,
                    'test_df': test_df_processed,
                    'initial_features': initial_features
                }, f)
            print(f"数据预处理完成，已保存至checkpoint")
            current_stage = 'initial_model'
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return None
    else:
        # Load preprocessed data from checkpoint
        try:
            print("从checkpoint加载预处理数据...")
            with open(preprocess_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
                train_df_processed = checkpoint_data['train_df']
                test_df_processed = checkpoint_data['test_df']
                initial_features = checkpoint_data['initial_features']
            print(f"已加载预处理数据，特征数量: {len(initial_features)}")
        except Exception as e:
            print(f"加载预处理数据失败: {e}")
            return None
    
    # Train initial model
    initial_pipeline = None
    initial_model = None
    initial_metrics = None
    feature_importance = None
    
    if current_stage in ['initial_model']:
        update_checkpoint_status('initial_model')
        print("\n训练初始模型...")
        try:
            initial_pipeline, initial_model, initial_metrics, feature_importance = train_funnel_model(
                train_df_processed, test_df_processed, target=target
            )
            
            # Save initial model and metrics for checkpoint
            with open(initial_model_file, 'wb') as f:
                pickle.dump({
                    'model': initial_model,
                    'metrics': initial_metrics,
                    'feature_importance': feature_importance
                }, f)
            print(f"初始模型训练完成，已保存checkpoint")
            current_stage = 'feature_analysis'
        except Exception as e:
            print(f"初始模型训练失败: {e}")
            return None
    elif current_stage not in ['start', 'preprocess']:
        # Load initial model from checkpoint
        try:
            print("从checkpoint加载初始模型...")
            with open(initial_model_file, 'rb') as f:
                model_data = pickle.load(f)
                initial_model = model_data['model']
                initial_metrics = model_data['metrics']
                feature_importance = model_data['feature_importance']
            print(f"已加载初始模型")
        except Exception as e:
            print(f"加载初始模型失败: {e}")
            return None
    
    # Analyze feature statistics and stability
    psi_results = None
    feature_stats = None
    must_include_features = []
    
    if current_stage in ['feature_analysis']:
        update_checkpoint_status('feature_analysis')
        try:
            # Analyze feature stability
            print("\n分析特征稳定性...")
            psi_results = analyze_feature_stability(train_df_processed, time_column='recall_date', n_bins=5)
            
            # Analyze feature statistics
            print("\n分析特征统计指标...")
            feature_stats = analyze_features_for_selection_parallel(
                train_df_processed, initial_features, target=target, n_jobs=4
            )
            
            # Find important features for positive sample prediction
            print("\n识别对正样本预测有重要作用的特征...")
            
            # Use SHAP values to find features with highest contribution to positive samples
            try:
                from joblib import load, dump
                import shap
                
                # Create small sample for SHAP analysis
                sample_size = min(1000, len(test_df_processed))
                sampled_indices = np.random.choice(len(test_df_processed), sample_size, replace=False)
                X_sample = test_df_processed.iloc[sampled_indices][initial_features]
                
                shap_file = os.path.join(MODEL_DIR, f"{target}_initial_shap.joblib")
                
                # Load cached SHAP values if available
                if os.path.exists(shap_file):
                    print(f"加载缓存的SHAP值: {shap_file}")
                    shap_values = load(shap_file)
                else:
                    print("计算SHAP值...")
                    explainer = shap.TreeExplainer(initial_model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Cache SHAP values
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    dump(shap_values, shap_file)
                    print(f"SHAP值已缓存至: {shap_file}")
                
                # Analyze SHAP values for positive samples
                pos_samples = test_df_processed[test_df_processed[target] == 1].index
                pos_indices = [i for i in range(len(sampled_indices)) if sampled_indices[i] in pos_samples]
                
                if pos_indices:
                    print(f"使用 {len(pos_indices)} 个正样本进行SHAP分析...")
                    pos_shap = np.abs(shap_values)[pos_indices].mean(axis=0)
                    pos_shap_dict = dict(zip(initial_features, pos_shap))
                    
                    # Get top SHAP features
                    top_shap_features = sorted(pos_shap_dict.items(), key=lambda x: x[1], reverse=True)[:30]
                    shap_features = [f for f, _ in top_shap_features]
                    
                    print("\n基于SHAP值对正样本预测最重要的前20个特征:")
                    for f, v in top_shap_features[:20]:
                        print(f"  {f}: {v:.6f}")
                    
                    must_include_features.extend(shap_features)
                    
                    # Perform positive sample subgroup analysis
                    print("\n执行正样本子群体分析...")
                    subgroup_features = analyze_positive_sample_subgroups(
                        train_df_processed, 
                        test_df_processed, 
                        initial_features, 
                        target, 
                        initial_model
                    )
                    
                    # Perform feature interaction analysis
                    print("\n执行特征交互分析...")
                    important_interactions = analyze_feature_interactions(
                        train_df_processed,
                        test_df_processed,
                        target,
                        initial_model,
                        initial_features
                    )
                    
                    # Extract individual features from interactions
                    interaction_features = []
                    for feat1, feat2 in important_interactions:
                        interaction_features.extend([feat1, feat2])
                    
                    # Add interaction features to must-include list
                    must_include_features.extend(interaction_features)
                    
                    # Add subgroup features to must-include list
                    must_include_features.extend(subgroup_features)
                    
                    # Ensure uniqueness of must-include features
                    must_include_features = list(set(must_include_features))
                    print(f"合并后必须包含的特征数量: {len(must_include_features)}")
            except Exception as e:
                print(f"SHAP分析出错: {e}")
            
            # Save feature analysis results
            with open(feature_analysis_file, 'wb') as f:
                pickle.dump({
                    'psi_results': psi_results,
                    'feature_stats': feature_stats,
                    'must_include_features': must_include_features
                }, f)
            print("特征分析完成，已保存checkpoint")
            current_stage = 'feature_selection'
        except Exception as e:
            print(f"特征分析失败: {e}")
            return None
    elif current_stage not in ['start', 'preprocess', 'initial_model']:
        # Load feature analysis from checkpoint
        try:
            print("从checkpoint加载特征分析结果...")
            with open(feature_analysis_file, 'rb') as f:
                analysis_data = pickle.load(f)
                psi_results = analysis_data['psi_results']
                feature_stats = analysis_data['feature_stats']
                must_include_features = analysis_data['must_include_features']
            print(f"已加载特征分析结果，必要特征数量: {len(must_include_features)}")
        except Exception as e:
            print(f"加载特征分析结果失败: {e}")
            return None
    
    # Feature selection step
    selected_features = None
    
    if current_stage in ['feature_selection']:
        update_checkpoint_status('feature_selection')
        # Define must-exclude features
        must_exclude_features = []
        
        # Define optimization weights
        weights = FEATURE_SELECTION_WEIGHTS
        
        # Second stage: trim features and retrain
        print("\n=== 第二阶段：基于筛选结果裁剪特征 ===")
        try:
            selected_features = trim_features_by_importance(
                initial_features,
                importance_dict=feature_importance,
                max_features=FEATURE_SELECTION_PARAMS['max_features'],
                train_df=train_df_processed,
                psi_results=psi_results,
                feature_stats=feature_stats,
                min_importance_pct=FEATURE_SELECTION_PARAMS['min_importance_pct'],
                max_psi=FEATURE_SELECTION_PARAMS['max_psi'],
                max_missing_rate=FEATURE_SELECTION_PARAMS['max_missing_rate'],
                min_iv=FEATURE_SELECTION_PARAMS['min_iv'],
                correlation_threshold=FEATURE_SELECTION_PARAMS['correlation_threshold'],
                must_include=must_include_features,
                must_exclude=must_exclude_features,
                weights=weights
            )
            
            print(f"筛选后特征数量: {len(selected_features)}")
            
            # Save selected feature list to checkpoint
            with open(selected_features_file, 'w') as f:
                json.dump({
                    'selected_features': selected_features
                }, f)
            
            # Save selected feature list for deployment
            feature_file_path = os.path.join(MODEL_DIR, f"{target}_selected_features.txt")
            save_feature_list(selected_features, feature_file_path)
            
            print("特征选择完成，已保存checkpoint")
            current_stage = 'final_model'
        except Exception as e:
            print(f"特征选择失败: {e}")
            return None
    elif current_stage not in ['start', 'preprocess', 'initial_model', 'feature_analysis']:
        # Load selected features from checkpoint
        try:
            print("从checkpoint加载已选特征...")
            with open(selected_features_file, 'r') as f:
                feature_data = json.load(f)
                selected_features = feature_data['selected_features']
            print(f"已加载已选特征，数量: {len(selected_features)}")
        except Exception as e:
            print(f"加载已选特征失败: {e}")
            return None
    
    # Final model training
    if current_stage in ['final_model']:
        update_checkpoint_status('final_model')
        print("\n=== 使用选定特征训练最终模型 ===")
        try:
            # Retrain with selected features
            train_df_final = preprocess_data(train_df.copy(), selected_features, keep_only_features=True)
            test_df_final = preprocess_data(test_df.copy(), selected_features, is_training=False, keep_only_features=True)
            
            # Second training with focus on positive sample prediction
            final_pipeline, final_model, final_metrics, _ = train_funnel_model(
                train_df_final, test_df_final, target=target
            )
            
            # Save final model using returned pipeline to preserve column names
            final_model_path = os.path.join(MODEL_DIR, f"{target}_final_model.pmml")
            sklearn2pmml(final_pipeline, final_model_path, with_repr=True)
            print(f"最终模型已保存至: {final_model_path}")
            
            result['final_model'] = final_model
            result['final_features'] = selected_features
            result['initial_metrics'] = initial_metrics
            result['final_metrics'] = final_metrics
            result['feature_file'] = os.path.join(MODEL_DIR, f"{target}_selected_features.txt")
            result['model_file'] = final_model_path
            
            current_stage = 'deployment'
        except Exception as e:
            print(f"最终模型训练失败: {e}")
            return None
    
    # Model deployment and validation
    if current_stage in ['deployment']:
        update_checkpoint_status('deployment')
        print("\n=== 模型部署验证 ===")
        try:
            feature_file_path = os.path.join(MODEL_DIR, f"{target}_selected_features.txt")
            final_model_path = os.path.join(MODEL_DIR, f"{target}_final_model.pmml")
            
            # Deployment validation - using exactly the same feature set as training
            deploy_results = deploy_model(
                final_model_path,
                test_df.copy(), 
                target, 
                feature_list_file=feature_file_path,
                output_dir=MODEL_DIR
            )
            
            result['deployment_metrics'] = deploy_results['metrics']
            
            # Compare initial and final model metrics
            print("\n模型指标比较:")
            print(f"{'指标':<15}{'初始模型':<15}{'最终模型':<15}{'部署验证':<15}")
            for metric in ['auc', 'pr_auc', 'ks', 'precision', 'recall', 'f1']:
                if metric in initial_metrics and metric in final_metrics and metric in deploy_results['metrics']:
                    initial_val = initial_metrics[metric]
                    final_val = final_metrics[metric]
                    deploy_val = deploy_results['metrics'][metric]
                    print(f"{metric:<15}{initial_val:<15.4f}{final_val:<15.4f}{deploy_val:<15.4f}")
            
            # Print final results and improvements
            if 'pr_auc' in initial_metrics and 'pr_auc' in final_metrics:
                pr_auc_improvement = final_metrics['pr_auc'] - initial_metrics['pr_auc']
                print(f"\nPR-AUC提升: {pr_auc_improvement:.4f} ({pr_auc_improvement/initial_metrics['pr_auc']*100:.2f}%)")
            
            if 'recall' in initial_metrics and 'recall' in final_metrics:
                recall_improvement = final_metrics['recall'] - initial_metrics['recall']
                print(f"正样本召回率提升: {recall_improvement:.4f} ({recall_improvement/initial_metrics['recall']*100:.2f}%)")
            
            # Mark pipeline as completed
            update_checkpoint_status('completed')
            print("\n模型训练与部署流程已完成")
        except Exception as e:
            print(f"模型部署验证失败: {e}")
            return None
    
    return result

# The deploy_model_with_selected_features function has been removed and replaced by the
# unified deploy_model function in src/deployment.py 