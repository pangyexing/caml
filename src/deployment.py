#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model deployment module.
Handles loading models, making predictions, and evaluating deployed models.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Import from shared modules
from src.model_utils import (
    load_pmml_model,
    get_model_features,
    prepare_prediction_data,
    predict_with_model
)
from src.evaluation import (
    create_score_bins,
    plot_score_distribution,
    evaluate_predictions,
    plot_precision_recall_curve
)
from src.preprocessing import preprocess_data
from src.utils import load_feature_list, save_model_metrics
from src.config import EXCLUDE_COLS, ID_COLS

def deploy_model(model_path: str,
                test_df: pd.DataFrame,
                target: str,
                feature_list_file: Optional[str] = None,
                output_dir: str = "deployment_results",
                bins: int = 10) -> Dict[str, Any]:
    """
    Deploy model and evaluate on test data.
    
    Args:
        model_path: Path to PMML model
        test_df: Test DataFrame
        target: Target variable name
        feature_list_file: Path to feature list file (optional)
        output_dir: Output directory
        bins: Score bin count
    
    Returns:
        Dictionary with deployment results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_pmml_model(model_path)
    
    # Get feature columns
    feature_cols = get_model_features(model, feature_list_file)
    if not feature_cols:
        if feature_list_file:
            feature_cols = load_feature_list(feature_list_file)
        if not feature_cols:
            raise ValueError("无法获取特征列，请提供有效的特征列表文件")
    
    print(f"部署使用 {len(feature_cols)} 个特征")
    
    # Filter out non-feature columns
    feature_cols = [col for col in feature_cols if col not in EXCLUDE_COLS]
    
    print(f"过滤后实际用于预测的特征数量: {len(feature_cols)}")
    
    # Preprocess test data
    test_df = preprocess_data(test_df, feature_cols, is_training=False, keep_only_features=True)
    
    # Prepare data for prediction
    X_test, ids_df = prepare_prediction_data(test_df, feature_cols)
    
    # Make predictions
    predictions = predict_with_model(model, X_test)
    
    # Add IDs if available
    if ids_df is not None:
        predictions = pd.concat([ids_df.reset_index(drop=True), predictions], axis=1)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, f"{target}_predictions.csv")
    predictions.to_csv(predictions_file, index=False)
    print(f"预测结果已保存至 {predictions_file}")
    
    # Save original index for alignment
    orig_index = test_df.index
    
    # Ensure prediction results align with original DataFrame
    if len(predictions) == len(test_df):
        # Reset prediction index to match test_df
        if not predictions.index.equals(orig_index):
            predictions.index = orig_index
        
        # Add prediction results to test_df
        test_df['score'] = predictions['score']
        test_df['prediction'] = predictions['prediction']
        test_df['probability'] = predictions['probability']
    else:
        # Handle potential row count mismatch
        print(f"警告: 预测结果数量({len(predictions)})与测试集行数({len(test_df)})不匹配")
        # Use pandas concat and reindex to ensure safe alignment
        for col in ['score', 'prediction', 'probability']:
            if col in predictions.columns:
                test_df[col] = pd.Series(predictions[col].values, index=orig_index)
    
    # Evaluate metrics
    metrics = {}
    bin_stats = None
    
    # Evaluate if target column exists
    if target in test_df.columns:
        y_true = test_df[target]
        metrics = evaluate_predictions(y_true, test_df['probability'])
        
        # Create score bins
        bin_stats = create_score_bins(
            test_df, 
            target_col=target, 
            output_file=os.path.join(output_dir, f"{target}_score_bins.csv")
        )
        
        # Plot score distribution
        plot_score_distribution(
            bin_stats, 
            target, 
            output_file=os.path.join(output_dir, f"{target}_score_distribution.png")
        )
        
        # Plot PR curve
        plot_precision_recall_curve(
            y_true, 
            test_df['probability'], 
            target, 
            output_file=os.path.join(output_dir, f"{target}_pr_curve.png")
        )
        
        # Export results with key columns
        export_cols = ['input_key', 'recall_date', target, 'score']
        if all(col in test_df.columns for col in export_cols):
            results_file = os.path.join(output_dir, f"{target}_deployment_results.csv")
            test_df[export_cols].to_csv(results_file, index=False)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"{target}_deployment_metrics.txt")
        save_model_metrics(metrics, f"Deployed {target} Model", metrics_file)
        
        # Print metrics summary
        print("\n模型部署验证指标:")
        print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"KS: {metrics['ks']:.4f}")
        print(f"最佳阈值: {metrics['threshold']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}, 精确率: {metrics['precision']:.4f}")
    
    # Return all results
    return {
        'model_path': model_path,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'bin_stats': bin_stats,
        'predictions_file': predictions_file,
        'results_dir': output_dir
    }

def batch_predict(model_path: str, 
                 input_file: str, 
                 feature_list_file: str, 
                 output_file: str, 
                 id_cols: List[str] = ID_COLS,
                 score_col: str = 'score',
                 threshold: float = 0.5) -> None:
    """
    Make batch predictions on input file.
    
    Args:
        model_path: Path to PMML model
        input_file: Input CSV file
        feature_list_file: Feature list file
        output_file: Output file for predictions
        id_cols: ID columns to include in output
        score_col: Column name for scores
        threshold: Classification threshold
    """
    # Load model
    model = load_pmml_model(model_path)
    
    # Load feature list
    feature_cols = load_feature_list(feature_list_file)
    
    # Load input data
    input_df = pd.read_csv(input_file)
    print(f"加载输入文件: {input_file}, 记录数: {len(input_df)}")
    
    # Check features exist in input
    missing_features = [f for f in feature_cols if f not in input_df.columns]
    if missing_features:
        print(f"警告: 输入文件中缺少以下特征: {missing_features}")
        print("将用0填充缺失特征")
        for feature in missing_features:
            input_df[feature] = 0.0
    
    # Preprocess data
    X, ids_df = prepare_prediction_data(input_df, feature_cols, id_cols)
    
    # Make predictions
    predictions = predict_with_model(model, X, threshold, score_col)
    
    # Combine with IDs
    if ids_df is not None:
        predictions = pd.concat([ids_df.reset_index(drop=True), predictions], axis=1)
    
    # Save predictions
    predictions.to_csv(output_file, index=False)
    print(f"批量预测结果已保存至 {output_file}")
    print(f"总记录数: {len(predictions)}")
    print(f"正例预测数: {predictions['prediction'].sum()}")
    
    return predictions 