#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training functionality.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

from src.core.config import DEFAULT_XGB_PARAMS, EXCLUDE_COLS, MODEL_DIR
from src.evaluation.metrics import evaluate_predictions
from src.features.analysis import (
    analyze_feature_interactions,
    analyze_positive_sample_subgroups,
)
from src.utils import configure_fonts_for_plots
from src.visualization.plots import (
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_summary,
)

# Call the font configuration function at module load time
configure_fonts_for_plots()

def train_initial_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = 'label_apply',
    features: Optional[List[str]] = None
) -> Tuple[PMMLPipeline, XGBClassifier, Dict[str, Any], Dict[str, float]]:
    """
    Train initial model for feature selection and evaluation.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target: Target label column name
        features: Feature list (if None, uses all non-excluded columns)
    
    Returns:
        pipeline: Trained PMML pipeline
        model: XGBoost model
        metrics: Evaluation metrics
        feature_importance: Feature importance dictionary
    """
    # Extract feature columns if not provided
    if features is None:
        features = [col for col in train_df.columns 
                  if col not in EXCLUDE_COLS and col != target]
    
    # Extract features and target
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Calculate class weights for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"训练集：正样本数量={pos_count}，负样本数量={neg_count}，正负样本比例=1:{neg_count/pos_count:.2f}")
    
    # Use default parameters for initial model
    params = DEFAULT_XGB_PARAMS.copy()
    params['scale_pos_weight'] = pos_weight
    params['eval_metric'] = 'aucpr'  # Move eval_metric to the parameters
    params['early_stopping_rounds'] = 20  # Move early_stopping_rounds to the parameters
    
    # Create model and pipeline
    xgb_clf = XGBClassifier(**params, verbosity=1)
    pipeline = PMMLPipeline([("classifier", xgb_clf)])

    # Train with early stopping
    print(f"开始训练初始模型 (特征数量: {len(features)})")
    eval_set = [(X_test, y_test)]
    pipeline.fit(X_train, y_train,
                classifier__eval_set=eval_set)

    # Extract model from pipeline
    model = pipeline.named_steps['classifier']

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate with standard metrics
    metrics = evaluate_predictions(
        y_test, 
        y_pred_proba, 
        threshold_method='f2',
        model_name='initial_model'
    )
    
    # Get feature importance
    feature_importance = model.get_booster().get_score(importance_type='gain')
    
    # Plot feature importance
    plot_feature_importance(
        feature_importance, 
        title=f'Initial Model Feature Importance for {target}',
        filepath=os.path.join(MODEL_DIR, "initial_model_importance.png")
    )
    
    # Generate precision-recall curve
    plot_precision_recall_curve(
        metrics['precision_curve'],
        metrics['recall_curve'],
        metrics['thresholds'],
        metrics['best_threshold'],
        metrics['f1_threshold'],
        metrics['pr_auc'],
        filepath=os.path.join(MODEL_DIR, "initial_model_pr_curve.png")
    )
    
    # Generate ROC curve
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['auc'],
        filepath=os.path.join(MODEL_DIR, "initial_model_roc_curve.png")
    )
    
    # Perform SHAP analysis
    try:
        perform_shap_analysis(
            model, 
            X_test, 
            'initial_model', 
            target
        )
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
    
    # Save model and pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(os.path.join(MODEL_DIR, "initial_model_importance.csv"), index=False)
    
    # Save model as PMML
    model_path = os.path.join(MODEL_DIR, "initial_model.pmml")
    sklearn2pmml(pipeline, model_path, with_repr=True)
    
    # Also save as pickle for easier Python reuse
    joblib.dump(model, os.path.join(MODEL_DIR, "initial_model.joblib"))
    
    return pipeline, model, metrics, feature_importance


def train_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str = 'label_apply',
    custom_params: Optional[Dict[str, Any]] = None
) -> Tuple[PMMLPipeline, XGBClassifier, Dict[str, Any]]:
    """
    Train final model with selected features.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        features: Selected feature list
        target: Target label column name
        custom_params: Custom XGBoost parameters (optional)
    
    Returns:
        pipeline: Trained PMML pipeline
        model: XGBoost model
        metrics: Evaluation metrics
    """
    # Extract features and target
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Calculate class weights for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"训练集：正样本数量={pos_count}，负样本数量={neg_count}，正负样本比例=1:{neg_count/pos_count:.2f}")
    
    # Use default parameters with custom overrides
    params = DEFAULT_XGB_PARAMS.copy()
    params['scale_pos_weight'] = pos_weight
    params['eval_metric'] = 'aucpr'  # Move eval_metric to the parameters
    params['early_stopping_rounds'] = 20  # Move early_stopping_rounds to the parameters
    
    if custom_params:
        params.update(custom_params)
    
    # Create model and pipeline
    xgb_clf = XGBClassifier(**params, verbosity=1)
    pipeline = PMMLPipeline([("classifier", xgb_clf)])

    # Train with early stopping
    print(f"开始训练最终模型 (特征数量: {len(features)})")
    eval_set = [(X_test, y_test)]
    pipeline.fit(X_train, y_train,
                classifier__eval_set=eval_set)

    # Extract model from pipeline
    model = pipeline.named_steps['classifier']

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate with standard metrics
    metrics = evaluate_predictions(
        y_test, 
        y_pred_proba, 
        threshold_method='f2',
        model_name='final_model'
    )
    
    # Get feature importance
    feature_importance = model.get_booster().get_score(importance_type='gain')
    
    # Plot feature importance
    plot_feature_importance(
        feature_importance, 
        title=f'Final Model Feature Importance for {target}',
        filepath=os.path.join(MODEL_DIR, "final_model_importance.png")
    )
    
    # Generate precision-recall curve
    plot_precision_recall_curve(
        metrics['precision_curve'],
        metrics['recall_curve'],
        metrics['thresholds'],
        metrics['best_threshold'],
        metrics['f1_threshold'],
        metrics['pr_auc'],
        filepath=os.path.join(MODEL_DIR, "final_model_pr_curve.png")
    )
    
    # Generate ROC curve
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['auc'],
        filepath=os.path.join(MODEL_DIR, "final_model_roc_curve.png")
    )
    
    # Perform SHAP analysis
    try:
        perform_shap_analysis(
            model, 
            X_test, 
            'final_model', 
            target
        )
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
    
    # Save model and pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(os.path.join(MODEL_DIR, "final_model_importance.csv"), index=False)
    
    # Save model as PMML
    model_path = os.path.join(MODEL_DIR, "final_model.pmml")
    sklearn2pmml(pipeline, model_path, with_repr=True)
    
    # Also save as pickle for easier Python reuse
    joblib.dump(model, os.path.join(MODEL_DIR, "final_model.joblib"))
    
    # Save selected features
    with open(os.path.join(MODEL_DIR, "selected_features.txt"), 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    return pipeline, model, metrics


def perform_shap_analysis(
    model: XGBClassifier,
    X: pd.DataFrame,
    model_name: str,
    target: str,
    max_samples: int = 1000
) -> None:
    """
    Perform SHAP analysis on the model.
    
    Args:
        model: Trained XGBoost model
        X: Feature DataFrame
        model_name: Model name for output files
        target: Target variable name
        max_samples: Maximum number of samples for SHAP analysis
    """
    print("执行 SHAP 特征分析...")
    
    # Sample data for performance
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Create and save summary plot
    plot_shap_summary(
        shap_values, 
        X_sample, 
        title=f'SHAP Feature Impact for {target}',
        filepath=os.path.join(MODEL_DIR, f"{model_name}_shap_summary.png")
    )
    
    # Create and save bar plot
    plot_shap_summary(
        shap_values, 
        X_sample, 
        title=f'SHAP Feature Importance for {target}',
        plot_type='bar',
        filepath=os.path.join(MODEL_DIR, f"{model_name}_shap_importance.png")
    )
    
    # Calculate and save average absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    shap_df.to_csv(os.path.join(MODEL_DIR, f"{model_name}_shap_importance.csv"), index=False)
    
    # Try to generate dependence plots for top 5 features
    try:
        top_features = shap_df['feature'].head(5).tolist()
        
        for feature in top_features:
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                feature, 
                shap_values, 
                X_sample, 
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_shap_dependence_{feature}.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Dependence plots failed: {str(e)}")


def two_stage_modeling_pipeline(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target: str = 'label_apply',
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete two-stage modeling pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target: Target label column name
        resume_from: Stage to resume from (if None, runs all stages)
    
    Returns:
        Dictionary with pipeline results
    """
    # Create a dictionary to store results
    pipeline_results = {
        'start_time': datetime.now(),
        'target': target,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'positive_train': train_df[target].sum(),
        'positive_test': test_df[target].sum(),
    }
    
    # Define pipeline stages
    stages = [
        'start',
        'initial_model',
        'feature_analysis',
        'feature_selection',
        'final_model',
        'completed'
    ]
    
    # Determine starting stage
    start_idx = 0
    if resume_from:
        try:
            start_idx = stages.index(resume_from)
            print(f"从阶段 '{resume_from}' 开始恢复")
        except ValueError:
            print(f"无效的恢复阶段 '{resume_from}'，从头开始")
    
    # Initialize checkpoint status tracker
    status = {stage: 'pending' for stage in stages}
    for stage in stages[:start_idx]:
        status[stage] = 'skipped'
    
    # Helper function to update status
    def update_status(stage, new_status='completed'):
        status[stage] = new_status
        pipeline_results['status'] = status
        # Print status update
        print(f"\n==== 阶段 '{stage}' {new_status} ====\n")
        
    # Update start status
    update_status('start')
    
    # STAGE 1: Initial model training
    if status['initial_model'] == 'pending':
        try:
            # Train initial model
            initial_pipeline, initial_model, initial_metrics, feature_importance = train_initial_model(
                train_df, test_df, target
            )
            
            # Save results
            pipeline_results['initial_model'] = {
                'metrics': initial_metrics,
                'feature_count': len(feature_importance),
                'model_path': os.path.join(MODEL_DIR, "initial_model.pmml")
            }
            
            update_status('initial_model')
        except Exception as e:
            print(f"初始模型训练失败: {str(e)}")
            update_status('initial_model', 'failed')
            return pipeline_results
    
    # STAGE 2: Feature analysis
    if status['feature_analysis'] == 'pending':
        try:
            # Extract models from previous stage
            initial_model = joblib.load(os.path.join(MODEL_DIR, "initial_model.joblib"))
            importance_df = pd.read_csv(os.path.join(MODEL_DIR, "initial_model_importance.csv"))
            feature_importance = dict(zip(importance_df['feature'], importance_df['importance']))
            
            # Get initial features
            initial_features = [col for col in train_df.columns if col not in EXCLUDE_COLS ]
            
            # Analyze positive sample subgroups
            subgroup_features = analyze_positive_sample_subgroups(
                train_df,
                test_df,
                initial_features,
                target,
                initial_model,
                n_clusters=3
            )
            
            # Analyze feature interactions
            interaction_pairs = analyze_feature_interactions(
                train_df,
                test_df,
                target,
                initial_model,
                initial_features,
                max_interactions=25
            )
            
            # Save results
            pipeline_results['feature_analysis'] = {
                'subgroup_count': len(subgroup_features),
                'interaction_count': len(interaction_pairs)
            }
            
            update_status('feature_analysis')
        except Exception as e:
            print(f"特征分析失败: {str(e)}")
            update_status('feature_analysis', 'failed')
            return pipeline_results
    
    # STAGE 3: Feature selection
    if status['feature_selection'] == 'pending':
        try:
            from src.features.analysis import analyze_features_for_selection_parallel
            from src.features.selection import trim_features_by_importance
            from src.features.stability import analyze_feature_stability
            from src.visualization.feature_selection_viz import visualize_feature_selection_results
            
            # Load previous results if needed
            importance_df = pd.read_csv(os.path.join(MODEL_DIR, "initial_model_importance.csv"))
            feature_importance = dict(zip(importance_df['feature'], importance_df['importance']))
            
            # Get initial features
            initial_features = list(feature_importance.keys())
            
            # Try to load subgroup features
            subgroup_features = []
            try:
                with open(os.path.join(MODEL_DIR, f"{target}_subgroup_features.txt"), 'r') as f:
                    subgroup_features = [line.strip() for line in f if line.strip()]
            except:
                print("未找到子群体特征文件，跳过子群体特征")
            
            # Analyze feature stability if time data available
            psi_results = None
            if 'recall_date' in train_df.columns:
                psi_results = analyze_feature_stability(
                    pd.concat([train_df, test_df]),
                    time_column='recall_date',
                    n_bins=5
                )
            
            # Analyze feature statistics
            feature_stats = analyze_features_for_selection_parallel(
                train_df,
                initial_features,
                target=target
            )
            
            # 从特征重要性中获取顶级特征
            top_model_feature_names = [f for f, _ in sorted(feature_importance.items(), 
                                                           key=lambda x: x[1], 
                                                           reverse=True)]
            
            # Create priority list - features that appear in both lists get priority
            shared_features = list(set(subgroup_features[:50]) & set(top_model_feature_names[:50]))
            # Add unique high-value features from each source
            remaining_features = list(set(subgroup_features[:30] + top_model_feature_names[:30]) - set(shared_features))
            # Combine features
            must_include = shared_features + remaining_features
            must_include = must_include[:50]  # Limit total
            
            # Run feature selection
            selected_features, feature_scores = trim_features_by_importance(
                initial_features,
                feature_importance,
                max_features=200,
                psi_results=psi_results,
                feature_stats=feature_stats,
                train_df=train_df,
                must_include=must_include,
                return_scores=True  # Return feature score details for visualization
            )
            
            # Visualize feature selection results
            visualize_feature_selection_results(
                feature_scores=feature_scores,
                feature_stats=feature_stats,
                selected_features=selected_features,
                train_df=train_df,
                output_dir=os.path.join(MODEL_DIR, 'feature_selection_viz')
            )
            
            # Save results
            pipeline_results['feature_selection'] = {
                'selected_count': len(selected_features),
                'initial_count': len(initial_features),
                'must_include_count': len(must_include),
                'visualization_path': os.path.join(MODEL_DIR, 'feature_selection_viz')
            }
            
            update_status('feature_selection')
        except Exception as e:
            print(f"特征选择失败: {str(e)}")
            update_status('feature_selection', 'failed')
            return pipeline_results
    
    # STAGE 4: Final model training
    if status['final_model'] == 'pending':
        try:
            # Load selected features
            selected_features = []
            with open(os.path.join(MODEL_DIR, "selected_features.txt"), 'r') as f:
                selected_features = [line.strip() for line in f if line.strip()]
            
            # Train final model
            final_pipeline, final_model, final_metrics = train_final_model(
                train_df,
                test_df,
                selected_features,
                target
            )
            
            # Save results
            pipeline_results['final_model'] = {
                'metrics': final_metrics,
                'feature_count': len(selected_features),
                'model_path': os.path.join(MODEL_DIR, "final_model.pmml")
            }
            
            update_status('final_model')
        except Exception as e:
            print(f"最终模型训练失败: {str(e)}")
            update_status('final_model', 'failed')
            return pipeline_results
    
    # Mark pipeline as completed
    update_status('completed')
    pipeline_results['end_time'] = datetime.now()
    pipeline_results['duration'] = (pipeline_results['end_time'] - pipeline_results['start_time']).total_seconds()
    
    # Print final summary
    print("\n==== 模型训练流程完成 ====")
    print(f"目标: {target}")
    print(f"总耗时: {pipeline_results['duration'] / 60:.2f} 分钟")
    
    if 'final_model' in pipeline_results:
        metrics = pipeline_results['final_model']['metrics'] 
        print("\n最终模型性能:")
        print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"最佳阈值: {metrics['best_threshold']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Save pipeline results
    output_path = os.path.join(MODEL_DIR, "pipeline_results.json")
    
    # Use the serialize_to_json utility from utils.common
    from src.utils.common import serialize_to_json
    serialize_to_json(pipeline_results, output_path)
    
    return pipeline_results 