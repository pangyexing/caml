#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for ML pipeline.
Contains common settings, parameters, and constants.
"""

import os
from typing import Dict, List, Any

# Directory paths
MODEL_DIR = "funnel_models"
TUNING_DIR = "tuned_models"
RESULTS_DIR = "optimization_results"
DEPLOYMENT_DIR = "deployment_results"

# Column definitions
# Essential ID columns
ID_COLS = ['input_key', 'recall_date']

# Label columns
LABEL_COLS = ['label_register', 'label_apply', 'label_approve']

# Non-feature columns to exclude
EXCLUDE_COLS = ID_COLS + LABEL_COLS + ['time_bin', 'customer_name', 'speaking_duration', 'label_intention', 'score']

# Default model parameters
DEFAULT_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 5,                   # Reduced tree depth to prevent overfitting
    'learning_rate': 0.05,            # Lower learning rate to better capture positive sample features
    'subsample': 0.85,                # Increased subsample ratio
    'colsample_bytree': 0.8,          # Feature column sampling ratio
    'min_child_weight': 2,            # Reduced minimum child weight to help capture minority class
    'gamma': 0.1,                     # Add regularization to control overfitting
    'reg_alpha': 0.01,                # L1 regularization
    'reg_lambda': 1.0,                # L2 regularization
    'use_label_encoder': False,
    'seed': 42
}

# Feature selection parameters
FEATURE_SELECTION_PARAMS = {
    'max_features': 200,
    'min_importance_pct': 0.005,     # Minimum feature importance percentage
    'max_psi': 0.3,                  # Maximum PSI (Population Stability Index)
    'max_missing_rate': 0.99,        # Maximum missing/zero rate
    'min_iv': 0.01,                  # Minimum Information Value
    'correlation_threshold': 0.95,   # Feature correlation threshold
}

# Feature selection weights
# Higher weights indicate more importance in selection decision
FEATURE_SELECTION_WEIGHTS = {
    'importance': 0.4,    # Model importance
    'psi': 0.1,           # Stability weight reduced
    'missing': 0.1,       # Missing rate
    'variance': 0.1,      # Variance
    'iv': 0.3             # IV weight increased (focus on target correlation)
}

# Hyperparameter tuning search space
HYPEROPT_SPACE = {
    'n_estimators': (50, 500, 50),   # (min, max, step)
    'max_depth': (3, 10, 1),         # (min, max, step)
    'learning_rate': (0.01, 0.3),    # (min, max) - log uniform
    'subsample': (0.6, 1.0),         # (min, max) - uniform
    'colsample_bytree': (0.6, 1.0),  # (min, max) - uniform
    'min_child_weight': (1, 10, 1),  # (min, max, step)
    'gamma': (0, 1),                 # (min, max) - uniform
    'reg_alpha': (1e-4, 1),          # (min, max) - log uniform
    'reg_lambda': (1e-4, 10),        # (min, max) - log uniform
}

# Model evaluation metrics
METRICS_LIST = ['auc', 'pr_auc', 'ks', 'precision', 'recall', 'f1']

# Default target variable
DEFAULT_TARGET = 'label_apply'

# Prediction settings
PREDICTION_SETTINGS = {
    'score_scale_factor': 1000,      # Scale factor for probability to score conversion
    'default_threshold': 0.5,        # Default classification threshold
    'score_bins': 10                 # Number of score bins for analysis
}

# Feature importance settings
IMPORTANCE_SETTINGS = {
    'top_n': 20,                    # Top N features to display in importance plots
    'importance_type': 'gain'       # Feature importance type (gain, weight, cover, total_gain, total_cover)
}

# Create necessary directories
def ensure_dirs():
    """Create all required directories if they don't exist."""
    for directory in [MODEL_DIR, TUNING_DIR, RESULTS_DIR, DEPLOYMENT_DIR]:
        os.makedirs(directory, exist_ok=True)
    return {
        'model_dir': MODEL_DIR,
        'tuning_dir': TUNING_DIR,
        'results_dir': RESULTS_DIR,
        'deployment_dir': DEPLOYMENT_DIR
    }

# Get model file paths
def get_model_paths(target: str = DEFAULT_TARGET) -> Dict[str, str]:
    """
    Get model file paths for a target.
    
    Args:
        target: Target variable name
        
    Returns:
        Dictionary with model file paths
    """
    return {
        'initial_model': os.path.join(MODEL_DIR, f"{target}_model.pmml"),
        'final_model': os.path.join(MODEL_DIR, f"{target}_final_model.pmml"),
        'tuned_model': os.path.join(TUNING_DIR, f"{target}_tuned_model.pmml"),
        'feature_list': os.path.join(MODEL_DIR, f"{target}_selected_features.txt"),
        'importance': os.path.join(MODEL_DIR, f"{target}_feature_importance.csv"),
        'shap_importance': os.path.join(MODEL_DIR, f"{target}_shap_importance.csv")
    } 