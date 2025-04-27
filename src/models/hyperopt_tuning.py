#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter optimization for XGBoost models.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.core.config import (
    HYPEROPT_PARAM_SPACE,
    HYPEROPT_SETTINGS,
    OPTIMIZATION_DIR,
    TUNED_MODELS_DIR,
)
from src.evaluation.metrics import evaluate_predictions
from src.models.training import train_final_model


# Add a custom JSON encoder class to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def create_hyperopt_space(param_space: Dict[str, List]) -> Dict[str, Any]:
    """
    Create a hyperopt search space from parameter lists.
    
    Args:
        param_space: Dictionary mapping parameter names to lists of possible values
        
    Returns:
        Hyperopt search space
    """
    space = {}
    
    for param, values in param_space.items():
        if param == 'max_depth':
            # max_depth should be an integer
            space[param] = scope.int(hp.quniform(param, min(values), max(values), 1))
        elif param == 'min_child_weight':
            # min_child_weight should be an integer
            space[param] = scope.int(hp.quniform(param, min(values), max(values), 1))
        elif all(isinstance(x, float) for x in values):
            # Continuous parameters
            space[param] = hp.uniform(param, min(values), max(values))
        else:
            # Categorical parameters
            space[param] = hp.choice(param, values)
    
    return space


def objective_xgb(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_folds: int = 3,
    random_state: int = 42,
    metric: str = 'pr_auc',
    threshold_method: str = 'f2'
) -> Dict[str, Any]:
    """
    Objective function for hyperopt optimization.
    
    Args:
        params: XGBoost parameters to evaluate
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        metric: Primary metric for optimization ('auc', 'pr_auc', etc.)
        threshold_method: Method for threshold optimization ('f1', 'f2')
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    
    # Handle integer parameters
    int_params = ['max_depth', 'min_child_weight']
    for param in int_params:
        if param in params:
            params[param] = int(params[param])
    
    # Add fixed parameters
    fixed_params = {
        'objective': 'binary:logistic',
        'seed': random_state
    }
    
    params.update(fixed_params)
    
    # Calculate pos_weight from training data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Cross-validation
    cv_scores = []
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in kf.split(X_train, y_train):
        X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_cv_train, y_cv_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_cv_train, y_cv_train)
        
        # Get best iteration if early stopping was used
        best_iteration = getattr(model, 'best_iteration', None)
        
        # Predict and evaluate
        y_cv_pred = model.predict_proba(X_cv_test)[:, 1]
        
        # Calculate metrics
        metrics = evaluate_predictions(
            y_cv_test, 
            y_cv_pred, 
            threshold_method=threshold_method,
            model_name='cv',
            output_dir=None  # Don't save files for CV
        )
        
        # Add metrics to list
        cv_scores.append({
            'auc': metrics['auc'],
            'pr_auc': metrics['pr_auc'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1'],
            'best_threshold': metrics['best_threshold']
        })
    
    # Calculate mean cross-validation scores
    mean_cv_scores = {k: np.mean([score[k] for score in cv_scores]) for k in cv_scores[0].keys()}
    
    # Evaluate on validation set
    val_model = XGBClassifier(**params)
    val_model.fit(X_train, y_train)
    
    # Get best iteration if early stopping was used
    best_val_iteration = getattr(val_model, 'best_iteration', None)
    
    # Predict and evaluate on validation set
    y_val_pred = val_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics on validation set
    val_metrics = evaluate_predictions(
        y_val, 
        y_val_pred, 
        threshold_method=threshold_method,
        model_name='val',
        output_dir=None  # Don't save files for validation
    )
    
    # Use the specified primary metric for optimization
    primary_metric_value = val_metrics.get(metric, 0.0)
    
    # Create result dictionary
    result = {
        'loss': -primary_metric_value,  # Hyperopt minimizes, so negate
        'status': STATUS_OK,
        'eval_time': time.time() - start_time,
        'params': params,
        'cv_scores': mean_cv_scores,
        'val_scores': {
            'auc': val_metrics['auc'],
            'pr_auc': val_metrics['pr_auc'],
            'recall': val_metrics['recall'],
            'precision': val_metrics['precision'],
            'f1': val_metrics['f1'],
            'best_threshold': val_metrics['best_threshold']
        }
    }
    
    return result


def hyperopt_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str = 'label_apply',
    max_evals: int = 50,
    param_space: Optional[Dict[str, List]] = None,
    metric: str = 'pr_auc',
    threshold_method: str = 'f2',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization for XGBoost using hyperopt.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        features: Features to use for training
        target: Target variable name
        max_evals: Maximum number of hyperopt evaluations
        param_space: Parameter space for hyperopt (or use default)
        metric: Primary metric for optimization
        threshold_method: Method for threshold optimization
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    print(f"\n开始超参数优化 (特征数量: {len(features)}, 最大评估次数: {max_evals})")
    
    # Create output directories
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    os.makedirs(TUNED_MODELS_DIR, exist_ok=True)
    
    # Use default parameter space if not provided
    if param_space is None:
        param_space = HYPEROPT_PARAM_SPACE
    
    # Create hyperopt search space
    space = create_hyperopt_space(param_space)
    
    # Create timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{target}_{timestamp}"
    
    # Split train into train and validation
    train_size = int(0.8 * len(train_df))
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_subset = train_df.iloc[:train_size]
    val_subset = train_df.iloc[train_size:]
    
    # Extract features and target
    X_train = train_subset[features]
    y_train = train_subset[target]
    X_val = val_subset[features]
    y_val = val_subset[target]
    
    # Create trials object to store results
    trials = Trials()
    
    # Define objective function with fixed data
    def objective(params):
        return objective_xgb(
            params,
            X_train, y_train,
            X_val, y_val,
            cv_folds=HYPEROPT_SETTINGS.get('cv_folds', 3),
            random_state=random_state,
            metric=metric,
            threshold_method=threshold_method
        )
    
    # Run hyperopt optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Convert best parameters for integer parameters
    best_params = {}
    for param, value in best.items():
        if param in ['max_depth', 'min_child_weight']:
            best_params[param] = int(value)
        elif param in param_space and isinstance(param_space[param], list) and not all(isinstance(x, float) for x in param_space[param]):
            # Handle choice parameters
            best_params[param] = param_space[param][value]
        else:
            best_params[param] = value
    
    # Get all results
    results = []
    for trial in trials.trials:
        if trial['result']['status'] == STATUS_OK:
            result = {
                'params': {k: v for k, v in trial['result']['params'].items() if k not in ['objective', 'use_label_encoder', 'seed']},
                'val_scores': trial['result']['val_scores'],
                'cv_scores': trial['result']['cv_scores'],
                'eval_time': trial['result']['eval_time']
            }
            results.append(result)
    
    # Sort results by validation metric
    results.sort(key=lambda x: x['val_scores'][metric], reverse=True)
    
    # Get top models
    top_models = results[:5]
    
    # Train best model on all training data
    print("\n训练最佳参数模型...")
    best_params_full = {
        'objective': 'binary:logistic',
        'seed': random_state
    }
    best_params_full.update(best_params)
    
    # Calculate pos_weight from full training data
    pos_count = train_df[target].sum()
    neg_count = len(train_df) - pos_count
    best_params_full['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Train final model with best parameters
    _, best_model, best_metrics = train_final_model(
        train_df,
        test_df,
        features,
        target=target,
        custom_params=best_params_full
    )
    
    # Save best model
    best_model_path = os.path.join(TUNED_MODELS_DIR, f"{run_name}_best_model.joblib")
    joblib.dump(best_model, best_model_path)
    
    # Create optimization report
    optimization_results = {
        'run_name': run_name,
        'target': target,
        'features': features,
        'feature_count': len(features),
        'max_evals': max_evals,
        'metric': metric,
        'threshold_method': threshold_method,
        'random_state': random_state,
        'start_time': start_time,
        'end_time': time.time(),
        'duration': time.time() - start_time,
        'best_params': best_params,
        'best_model_path': best_model_path,
        'best_model_metrics': best_metrics,
        'top_models': top_models,
        'param_space': param_space
    }
    
    # Save optimization results
    results_path = os.path.join(OPTIMIZATION_DIR, f"{run_name}_results.json")
    
    # Convert to JSON-serializable format
    json_results = {
        'run_name': run_name,
        'target': target,
        'feature_count': len(features),
        'max_evals': max_evals,
        'metric': metric,
        'threshold_method': threshold_method,
        'random_state': random_state,
        'start_time': str(datetime.fromtimestamp(start_time)),
        'end_time': str(datetime.now()),
        'duration': time.time() - start_time,
        'best_params': best_params,
        'best_model_path': best_model_path,
        'best_model_metrics': {k: v for k, v in best_metrics.items() if not isinstance(v, np.ndarray)},
        'top_models': top_models,
        'param_space': param_space
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, cls=NumpyEncoder)
    
    # Create and save parameter importance visualization
    plt.figure(figsize=(12, 8))
    
    # Extract parameter values and corresponding performance
    param_importance = {}
    for param in best_params.keys():
        if param not in ['objective', 'use_label_encoder', 'seed', 'scale_pos_weight']:
            param_values = []
            param_scores = []
            
            for result in results:
                if param in result['params']:
                    param_values.append(result['params'][param])
                    param_scores.append(result['val_scores'][metric])
            
            if param_values:
                param_importance[param] = np.corrcoef(param_values, param_scores)[0, 1]
    
    # Plot parameter importance
    params = list(param_importance.keys())
    importances = [abs(param_importance[p]) for p in params]
    
    plt.barh(params, importances)
    plt.xlabel('Absolute Correlation with Performance')
    plt.title('Parameter Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OPTIMIZATION_DIR, f"{run_name}_parameter_importance.png"))
    plt.close()
    
    # Print optimization summary
    print(f"\n超参数优化完成，共评估 {len(results)} 组参数，耗时 {time.time() - start_time:.2f} 秒")
    print("\n最佳参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n最佳模型性能 ({metric}):")
    for metric_name, value in best_metrics.items():
        if not isinstance(value, np.ndarray):
            print(f"  {metric_name}: {value}")
    
    return optimization_results


def plot_optimization_results(results_path: str) -> None:
    """
    Plot optimization results from a saved results file.
    
    Args:
        results_path: Path to the optimization results JSON file
    """
    # Check if file exists
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
        
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    top_models = results.get('top_models', [])
    if not top_models:
        print("No results to plot")
        return
    
    metric = results.get('metric', 'pr_auc')
    
    # Create figure for parameter vs. performance
    plt.figure(figsize=(15, 10))
    
    # Get list of parameters
    params = set()
    for model in top_models:
        params.update(model['params'].keys())
    
    params = sorted(list(params))
    
    # Create subplots for each parameter
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    for i, param in enumerate(params):
        plt.subplot(n_rows, n_cols, i+1)
        
        param_values = []
        metric_values = []
        
        for model in top_models:
            if param in model['params']:
                param_values.append(model['params'][param])
                metric_values.append(model['val_scores'][metric])
        
        plt.scatter(param_values, metric_values, alpha=0.6)
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.title(f"{param} vs. {metric}")
        
        # Try to highlight the best value
        if param in results.get('best_params', {}):
            best_value = results['best_params'][param]
            plt.axvline(best_value, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(results_path), 
                              f"{results['run_name']}_parameter_plots.png")
    plt.savefig(output_path)
    plt.close()
    
    # Create plot of top models
    plt.figure(figsize=(12, 8))
    
    # Sort models by performance
    top_models.sort(key=lambda x: x['val_scores'][metric], reverse=True)
    
    # Plot top models performance
    metrics_to_plot = ['auc', 'pr_auc', 'recall', 'precision', 'f1']
    metrics_available = []
    
    for m in metrics_to_plot:
        if all(m in model['val_scores'] for model in top_models):
            metrics_available.append(m)
    
    # Create bar chart
    model_indices = list(range(len(top_models)))
    bar_width = 0.15
    offset = -(len(metrics_available) - 1) * bar_width / 2
    
    for i, m in enumerate(metrics_available):
        values = [model['val_scores'][m] for model in top_models]
        plt.bar([x + offset + i * bar_width for x in model_indices], 
                values, width=bar_width, label=m)
    
    plt.xlabel('Model Rank')
    plt.ylabel('Score')
    plt.title('Top Models Performance')
    plt.xticks(model_indices, [f"Model {i+1}" for i in model_indices])
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(results_path), 
                              f"{results['run_name']}_models_comparison.png")
    plt.savefig(output_path)
    plt.close()


def get_best_params_from_results(results_path: str) -> Dict[str, Any]:
    """
    Extract best parameters from an optimization results file.
    
    Args:
        results_path: Path to the optimization results JSON file
        
    Returns:
        Dictionary of best parameters
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract best parameters
    best_params = results.get('best_params', {})
    
    # Add required parameters
    best_params.update({
        'objective': 'binary:logistic',
        'seed': results.get('random_state', 42)
    })
    
    return best_params 