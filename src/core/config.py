#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters and constants for the ML pipeline.
"""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "funnel_models")
TUNED_MODELS_DIR = os.path.join(BASE_DIR, "tuned_models")
OPTIMIZATION_DIR = os.path.join(BASE_DIR, "optimization_results")
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment_results")

# Create directories if they don't exist
for directory in [MODEL_DIR, TUNED_MODELS_DIR, OPTIMIZATION_DIR, DEPLOYMENT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Column exclusions for feature selection
EXCLUDE_COLS = [
    'input_key', 'recall_date', 'label_register', 
    'label_apply', 'label_approve', 'time_bin', 'score'
]

# Feature selection parameters
FEATURE_SELECTION_PARAMS = {
    'min_importance_pct': 0.005,  # Minimum relative importance
    'max_psi': 0.25,              # Maximum PSI for stability
    'max_missing_rate': 0.99,     # Maximum missing rate
    'min_variance': 1e-6,         # Minimum variance threshold
    'min_iv': 0.02,               # Minimum information value
    'correlation_threshold': 0.9, # Correlation threshold for feature removal
}

# Weights for feature selection scoring
FEATURE_SELECTION_WEIGHTS = {
    'importance': 0.4,  # Model importance
    'psi': 0.2,         # Stability 
    'missing': 0.1,     # Missing rate
    'variance': 0.1,    # Variance
    'iv': 0.2           # Information value
}

# Default XGBoost parameters
DEFAULT_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 5,                
    'learning_rate': 0.05,         
    'subsample': 0.85,             
    'colsample_bytree': 0.8,       
    'min_child_weight': 2,         
    'gamma': 0.1,                 
    'reg_alpha': 0.01,             
    'reg_lambda': 1.0,             
    'use_label_encoder': False,
    'seed': 42
}

# Hyperparameter search space
HYPEROPT_PARAM_SPACE = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
    'min_child_weight': [1, 2, 3, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
}

# Hyperparameter optimization settings
HYPEROPT_SETTINGS = {
    'max_evals': 50,           # Maximum evaluations
    'cv_folds': 3,             # Cross-validation folds
    'metric': 'pr_auc',        # Primary optimization metric
    'secondary_metrics': ['recall_at_threshold', 'f1_at_threshold'],
    'threshold_method': 'f2',  # F2 favors recall more
    'early_stopping_rounds': 20,
    'random_state': 42
}

# SHAP analysis settings
SHAP_SETTINGS = {
    'max_samples': 1000,       # Maximum samples for SHAP analysis
    'max_display': 20,         # Maximum features to display in SHAP plots
    'skip_interactions_if_samples_over': 5000  # Skip interaction analysis for large datasets
}

# Language settings
LANGUAGE = 'zh'  # 'zh' for Chinese, 'en' for English

# Translation dictionary for output messages
TRANSLATIONS = {
    'model_training_started': {
        'en': 'Model training started for target: {}',
        'zh': '开始训练模型，目标变量: {}'
    },
    'feature_selection_complete': {
        'en': 'Feature selection complete. Selected {} features.',
        'zh': '特征选择完成，选择了 {} 个特征。'
    }
}

def get_message(key: str, lang: str = None, *args) -> str:
    """Get localized message with formatting."""
    if lang is None:
        lang = LANGUAGE
    
    if key in TRANSLATIONS:
        msg_template = TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get('en', key))
        return msg_template.format(*args) if args else msg_template
    return key 