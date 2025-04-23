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
    save_feature_list
)
# Import configuration module
from src.config import FEATURE_SELECTION_PARAMS, FEATURE_SELECTION_WEIGHTS

def train_funnel_model(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      target: str = 'label_apply') -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, float]]:
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
    model_dir = "funnel_models"
    os.makedirs(model_dir, exist_ok=True)
    importance_file = os.path.join(model_dir, f"{target}_feature_importance.csv")
    
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
    model_dir = "funnel_models"
    shap_importance_file = os.path.join(model_dir, f"{target}_shap_importance.csv")
    
    shap_items = sorted(shap_importance_dict.items(), key=lambda item: item[1], reverse=True)
    shap_df = pd.DataFrame(shap_items, columns=['feature', 'importance'])
    shap_df.to_csv(shap_importance_file, index=False)

    # Save model as PMML file
    model_path = os.path.join(model_dir, f"{target}_model.pmml")
    sklearn2pmml(pipeline, model_path, with_repr=True)
    
    return model, metrics, feature_importance

def two_stage_modeling_pipeline(train_df: pd.DataFrame, 
                               test_df: pd.DataFrame, 
                               target: str = 'label_apply') -> Dict[str, Any]:
    """
    Two-stage modeling pipeline: first train an initial model for feature selection,
    then retrain with selected features and deploy.
    Optimization goal: improve positive sample prediction and PR-AUC metric.
    
    Args:
        train_df: Training dataset
        test_df: Testing dataset
        target: Target variable
    
    Returns:
        Dictionary with final model, features, and evaluation metrics
    """
    # First stage: train initial model with all features
    print("=== 第一阶段：使用全量特征训练初始模型 ===")
    exclude_cols = ['input_key', 'recall_date', 'label_register', 'label_apply', 'label_approve', 'time_bin', 'score']
    initial_features = [col for col in train_df.columns if col not in exclude_cols]
    print(f"初始特征数量: {len(initial_features)}")
    
    # Analyze sample distribution
    pos_count = train_df[target].sum()
    total_count = len(train_df)
    pos_ratio = pos_count / total_count
    print(f"训练集正样本比例: {pos_count}/{total_count} = {pos_ratio:.2%}")
    
    # Preprocess training data
    train_df_processed = preprocess_data(train_df.copy(), initial_features)
    test_df_processed = preprocess_data(test_df.copy(), initial_features, is_training=False)
    
    # Train initial model
    print("\n训练初始模型...")
    initial_model, initial_metrics, feature_importance = train_funnel_model(
        train_df_processed, test_df_processed, target=target
    )
    
    # Analyze feature stability
    print("\n分析特征稳定性...")
    psi_results = analyze_feature_stability(train_df_processed, time_column='recall_date', n_bins=5)
    
    # Analyze feature statistics
    print("\n分析特征统计指标...")
    feature_stats = analyze_features_for_selection_parallel(
        train_df_processed, initial_features, target=target, n_jobs=4
    )
    
    # Define must-include features
    must_include_features = []
    
    # Find important features for positive sample prediction
    print("\n识别对正样本预测有重要作用的特征...")
    
    # Use SHAP values to find features with highest contribution to positive samples
    try:
        from joblib import load, dump
        import shap
        import os
        
        # Create small sample for SHAP analysis
        sample_size = min(1000, len(test_df_processed))
        sampled_indices = np.random.choice(len(test_df_processed), sample_size, replace=False)
        X_sample = test_df_processed.iloc[sampled_indices][initial_features]
        
        shap_file = os.path.join("funnel_models", f"{target}_initial_shap.joblib")
        
        # Load cached SHAP values if available
        if os.path.exists(shap_file):
            print(f"加载缓存的SHAP值: {shap_file}")
            shap_values = load(shap_file)
        else:
            print("计算SHAP值...")
            explainer = shap.TreeExplainer(initial_model)
            shap_values = explainer.shap_values(X_sample)
            
            # Cache SHAP values
            os.makedirs("funnel_models", exist_ok=True)
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
    except Exception as e:
        print(f"SHAP分析出错: {e}")
    
    # Define must-exclude features
    must_exclude_features = []
    
    # Define optimization weights
    weights = FEATURE_SELECTION_WEIGHTS
    
    # Second stage: trim features and retrain
    print("\n=== 第二阶段：基于筛选结果裁剪特征并重新训练 ===")
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
    
    # Save selected feature list
    model_dir = "funnel_models"
    os.makedirs(model_dir, exist_ok=True)
    feature_file_path = os.path.join(model_dir, f"{target}_selected_features.txt")
    
    # Use shared function to save feature list
    save_feature_list(selected_features, feature_file_path)
    
    # Retrain with selected features
    train_df_final = preprocess_data(train_df.copy(), selected_features, keep_only_features=True)
    test_df_final = preprocess_data(test_df.copy(), selected_features, is_training=False, keep_only_features=True)
    
    # Second training with focus on positive sample prediction
    final_model, final_metrics, _ = train_funnel_model(
        train_df_final, test_df_final, target=target
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"{target}_final_model.pmml")
    pipeline = PMMLPipeline([("classifier", final_model)])
    sklearn2pmml(pipeline, final_model_path, with_repr=True)
    
    print("\n=== 模型部署验证 ===")
    # Deployment validation - using exactly the same feature set as training
    deploy_results = deploy_model_with_selected_features(
        test_df.copy(), 
        target, 
        final_model_path,
        feature_list_file=feature_file_path
    )
    
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
    
    return {
        'final_model': final_model,
        'final_features': selected_features,
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'deployment_metrics': deploy_results['metrics'],
        'feature_file': feature_file_path,
        'model_file': final_model_path
    }

def deploy_model_with_selected_features(test_df: pd.DataFrame, 
                                       target: str, 
                                       model_path: str, 
                                       feature_list_file: str, 
                                       bins: int = 10) -> Dict[str, Any]:
    """
    Simplified model deployment function using the same feature set as training.
    
    Args:
        test_df: Test dataset
        target: Target variable
        model_path: PMML model path
        feature_list_file: Feature list file path
        bins: Score bin count
    
    Returns:
        Dictionary with deployment metrics and results
    """
    # Load model and feature list
    model = load_pmml_model(model_path)
    feature_cols = load_feature_list(feature_list_file)
    
    print(f"部署使用 {len(feature_cols)} 个特征 (与第二次训练完全一致)")
    
    # Filter out non-feature columns
    exclude_cols = ['input_key', 'recall_date', 'label_register', 'label_apply', 'label_approve', 'time_bin', 'score']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    print(f"过滤后实际用于预测的特征数量: {len(feature_cols)}")
    
    # Preprocess test data
    test_df = preprocess_data(test_df, feature_cols, is_training=False, keep_only_features=True)
    
    # Prepare data for prediction
    X_test, _ = prepare_prediction_data(test_df, feature_cols)
    
    # Make predictions
    predictions = predict_with_model(model, X_test)
    
    # Add predictions to test data
    test_df['score'] = predictions['score']
    test_df['prediction'] = predictions['prediction']
    test_df['probability'] = predictions['probability']
    
    # Create score bins
    bin_stats = create_score_bins(
        test_df, 
        target_col=target, 
        output_file=f"{target}_score_bins.csv"
    )
    
    # Plot score distribution
    plot_score_distribution(bin_stats, target)
    
    # Calculate metrics
    metrics = evaluate_predictions(test_df[target], test_df['probability'])
    
    # Export results
    export_cols = ['input_key', 'recall_date', target, 'score']
    if all(col in test_df.columns for col in export_cols):
        test_df[export_cols].to_csv(f"{target}_deployment_results.csv", index=False)
    
    # Print metrics
    print("\n模型部署验证指标:")
    print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"KS: {metrics['ks']:.4f}")
    print(f"最佳阈值: {metrics['threshold']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}, 精确率: {metrics['precision']:.4f}")
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'bin_stats': bin_stats,
        'results_file': f"{target}_deployment_results.csv"
    } 