#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model deployment functionality.
"""

import os
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pypmml import Model as PyPMMLModel

from src.core.config import DEPLOYMENT_DIR, ID_COLS
from src.evaluation.metrics import calculate_lift, plot_lift_chart
from src.visualization.plots import create_score_bins, plot_score_distribution
from src.utils.common import serialize_to_json


def load_model(model_path: str):
    """
    Load a model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    if model_path.endswith('.pmml'):
        # Load PMML model using pypmml
        model = PyPMMLModel.load(model_path)
    elif model_path.endswith('.joblib'):
        # Load joblib model
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    
    return model


def get_model_features(model) -> List[str]:
    """
    Get the list of features the model was trained on.
    
    Args:
        model: Trained model
        
    Returns:
        List of feature names
    """
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
        feature_names = model.get_booster().feature_names
        if feature_names is not None:
            return feature_names
    
    # For PMML models
    if hasattr(model, 'inputFields'):
        return [field.name for field in model.inputFields]
    
    raise ValueError("Could not retrieve feature names from model")


def prepare_prediction_data(
    df: pd.DataFrame, 
    feature_list: List[str]
) -> pd.DataFrame:
    """
    Prepare data for prediction by ensuring all required features are present.
    
    Args:
        df: Input DataFrame
        feature_list: List of features required by the model
        
    Returns:
        DataFrame with all required features
    """
    # Check for missing features
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features for prediction: {missing_features}")
    
    # Select only required features
    prediction_df = df[feature_list].copy()
    
    # Handle missing values
    for col in prediction_df.columns:
        if prediction_df[col].isna().any():
            if prediction_df[col].dtype in ['float64', 'int64']:
                # For numeric columns, fill with median
                prediction_df[col] = prediction_df[col].fillna(0.0)
            else:
                # For categorical columns, fill with most frequent value
                most_common = prediction_df[col].mode()[0] if not prediction_df[col].mode().empty else 0
                prediction_df[col] = prediction_df[col].fillna(most_common)
    
    return prediction_df


def predict_with_model(model, df: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with a model.
    
    Args:
        model: Trained model
        df: Input DataFrame
        threshold: Classification threshold
        
    Returns:
        Tuple of (probability predictions, binary predictions)
    """
    # Check model type and call appropriate prediction method
    if hasattr(model, 'predict_proba'):
        # scikit-learn style model
        y_pred_proba = model.predict_proba(df)[:, 1]
    elif hasattr(model, 'predict'):
        # PMML model
        if isinstance(model, PyPMMLModel):
            # PyPMML model prediction
            predictions = []
            for _, row in df.iterrows():
                result = model.predict(row.to_dict())
                score = result.get('probability(1)') or result.get('probability_1') or result.get('probability')
                if score is None and 'predicted' in result:
                    # For models that only return class labels
                    score = float(result['predicted'])
                predictions.append(score)
            y_pred_proba = np.array(predictions)
        else:
            # Generic model with predict method
            y_pred_proba = model.predict(df)
    else:
        raise ValueError("Model does not have predict_proba or predict method")
    
    # Convert to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred_proba, y_pred


def save_feature_list(features: List[str], output_path: str) -> None:
    """
    Save a list of features to a file.
    
    Args:
        features: List of feature names
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")


def load_feature_list(file_path: str) -> List[str]:
    """
    Load a list of features from a file.
    
    Args:
        file_path: Path to the feature list file
        
    Returns:
        List of feature names
    """
    with open(file_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def deploy_model(
    model_path: str,
    test_df: pd.DataFrame,
    target: Optional[str] = None,
    features_file: Optional[str] = None,
    threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a model and evaluate it on test data.
    
    Args:
        model_path: Path to the model file
        test_df: Test DataFrame
        target: Target column name (if available for evaluation)
        features_file: Path to file containing feature list (if model doesn't provide features)
        threshold: Classification threshold (if None, use default or find optimal)
        output_dir: Output directory for deployment results
        deployment_name: Name for this deployment
        
    Returns:
        Dictionary with deployment results
    """
    start_time = time.time()
    
    # Create timestamp for deployment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not deployment_name:
        deployment_name = f"deployment_{timestamp}"
    
    # Set output directory
    if not output_dir:
        output_dir = os.path.join(DEPLOYMENT_DIR, deployment_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始模型部署: {deployment_name}")
    print(f"模型路径: {model_path}")
    print(f"测试数据大小: {len(test_df)} 行")
    
    # Load model
    model = load_model(model_path)
    
    # Get features for prediction
    if features_file:
        features = load_feature_list(features_file)
        print(f"从文件加载特征列表，共 {len(features)} 个特征")
    else:
        raise ValueError("Could not get features from model and no features file provided")

    # try:
    #     features = get_model_features(model)
    #     print(f"从模型获取特征列表，共 {len(features)} 个特征")
    # except ValueError:
    #     if features_file:
    #         features = load_feature_list(features_file)
    #         print(f"从文件加载特征列表，共 {len(features)} 个特征")
    #     else:
    #         raise ValueError("Could not get features from model and no features file provided")
    
    # Save feature list
    feature_list_path = os.path.join(output_dir, "model_features.txt")
    save_feature_list(features, feature_list_path)
    
    # Prepare prediction data
    X_test = prepare_prediction_data(test_df, features)
    
    # Make predictions
    y_pred_proba, _ = predict_with_model(model, X_test, threshold=0.5)
    
    # Evaluate if target is available
    deployment_results = {
        'model_path': model_path,
        'feature_count': len(features),
        'test_count': len(test_df),
        'deployment_name': deployment_name,
        'timestamp': timestamp,
        'feature_list_path': feature_list_path
    }
    
    # Save scores to DataFrame
    scores_df = pd.DataFrame({
        'score': y_pred_proba
    })
    
    # Add ID columns from test_df to scores_df
    for col in ID_COLS:
        if col in test_df.columns:
            scores_df[col] = test_df[col]
    
    # Add feature columns to scores_df
    for feature in features:
        if feature in test_df.columns:
            scores_df[feature] = test_df[feature]
    
    if target and target in test_df.columns:
        from src.evaluation.metrics import evaluate_predictions
        
        # Use provided threshold or find optimal
        if threshold is None:
            # Evaluate with default F2 optimization
            metrics = evaluate_predictions(
                test_df[target].values,
                y_pred_proba,
                threshold_method='f2',
                model_name=deployment_name,
                output_dir=output_dir
            )
            threshold = metrics['best_threshold']
        else:
            # Evaluate with provided threshold
            metrics = evaluate_predictions(
                test_df[target].values,
                y_pred_proba,
                threshold_method='custom',
                custom_threshold=threshold,
                model_name=deployment_name,
                output_dir=output_dir
            )
        
        # Calculate lift
        lift_df = calculate_lift(test_df[target].values, y_pred_proba)
        lift_df.to_csv(os.path.join(output_dir, f"{deployment_name}_lift.csv"), index=False)
        
        # Plot lift chart
        plot_lift_chart(lift_df, os.path.join(output_dir, f"{deployment_name}_lift_chart.png"))
        
        # Generate score distribution
        plot_score_distribution(
            y_pred_proba,
            test_df[target].values,
            title=f'Score Distribution for {deployment_name}',
            filepath=os.path.join(output_dir, f"{deployment_name}_score_distribution.png")
        )
        
        # Add evaluation results to deployment results
        deployment_results.update({
            'has_target': True,
            'target_column': target,
            'positive_count': test_df[target].sum(),
            'positive_rate': test_df[target].mean(),
            'threshold': threshold,
            'metrics': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        })
        
        # Add target to scores DataFrame
        scores_df['target'] = test_df[target]
        
        # Make binary predictions with selected threshold
        scores_df['prediction'] = (scores_df['score'] >= threshold).astype(int)
        
        # Print evaluation summary
        print(f"\n模型评估结果 (阈值={threshold:.4f}):")
        print(f"AUC: {metrics['auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
    else:
        print("\n无目标变量，跳过模型评估")
        
        # Just create score bins and distribution without target
        bins, bin_indices, bin_labels = create_score_bins(y_pred_proba)
        
        bin_df = pd.DataFrame({
            'bin_label': bin_labels,
            'min_score': bins[:-1],
            'max_score': bins[1:],
            'count': np.bincount(bin_indices, minlength=len(bin_labels))
        })
        
        bin_df.to_csv(os.path.join(output_dir, f"{deployment_name}_score_bins.csv"), index=False)
        
        # Use default threshold if not provided
        threshold = 0.5 if threshold is None else threshold
        
        # Make binary predictions with selected threshold
        scores_df['prediction'] = (scores_df['score'] >= threshold).astype(int)
        
        # Add to deployment results
        deployment_results.update({
            'has_target': False,
            'threshold': threshold
        })
    
    # Save prediction scores
    scores_path = os.path.join(output_dir, f"{deployment_name}_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    deployment_results['scores_path'] = scores_path
    
    # Calculate statistics at different score thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_stats = []
    
    for t in thresholds:
        predicted_pos = (y_pred_proba >= t).sum()
        threshold_stats.append({
            'threshold': t,
            'predicted_positive': predicted_pos,
            'predicted_positive_rate': predicted_pos / len(test_df)
        })
    
    # Save threshold statistics
    threshold_df = pd.DataFrame(threshold_stats)
    threshold_df.to_csv(os.path.join(output_dir, f"{deployment_name}_thresholds.csv"), index=False)
    
    # Save deployment results
    deployment_results['duration'] = time.time() - start_time
    
    results_path = os.path.join(output_dir, f"{deployment_name}_results.json")
    
    serialize_to_json(deployment_results, results_path)
    
    print(f"\n部署完成，耗时 {time.time() - start_time:.2f} 秒")
    print(f"结果保存在: {output_dir}")
    
    return deployment_results


def batch_prediction(
    model_path: str,
    input_df: pd.DataFrame,
    key_column: str,
    features_file: str,
    output_file: str,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Run batch prediction with a model and save results.
    
    Args:
        model_path: Path to the model file
        input_df: Input DataFrame
        key_column: Name of the key column to include in results
        features_file: Path to file containing feature list
        output_file: Path to save prediction results
        threshold: Classification threshold
        
    Returns:
        DataFrame with predictions
    """
    print(f"开始批量预测，数据量: {len(input_df)} 行")
    
    # Load model
    model = load_model(model_path)
    
    # Load features
    features = load_feature_list(features_file)
    print(f"特征数量: {len(features)}")
    
    # Prepare prediction data (handling missing columns)
    missing_cols = [col for col in features if col not in input_df.columns]
    if missing_cols:
        print(f"警告: 缺少 {len(missing_cols)} 个特征列，将使用0填充")
        for col in missing_cols:
            input_df[col] = 0.0
    
    X = prepare_prediction_data(input_df, features)
    
    # Make predictions
    y_pred_proba, y_pred = predict_with_model(model, X, threshold)
    
    # Create result DataFrame
    if key_column in input_df.columns:
        result_df = pd.DataFrame({
            key_column: input_df[key_column],
            'score': y_pred_proba,
            'prediction': y_pred
        })
    else:
        result_df = pd.DataFrame({
            'score': y_pred_proba,
            'prediction': y_pred
        })
    
    # Add ID columns from input_df to result_df
    for col in ID_COLS:
        if col in input_df.columns and col != key_column:
            result_df[col] = input_df[col]
    
    # Add feature columns to result_df
    for feature in features:
        if feature in input_df.columns:
            result_df[feature] = input_df[feature]
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    print(f"预测完成，结果保存在: {output_file}")
    print(f"预测为正 (>= {threshold}) 的样本数: {y_pred.sum()} ({y_pred.mean():.2%})")
    
    return result_df 