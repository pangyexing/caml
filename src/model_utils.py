#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model utilities module.
Shared functions for model loading, feature handling, and prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pypmml import Model

# Import local modules
from src.preprocessing import preprocess_data
from src.utils import load_feature_list
from src.config import EXCLUDE_COLS, ID_COLS

def load_pmml_model(model_path: str) -> Model:
    """
    Load a PMML model.
    
    Args:
        model_path: Path to PMML model file
        
    Returns:
        Loaded PMML model
    """
    try:
        model = Model.load(model_path)
        print(f"已加载PMML模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        raise

def get_model_features(model: Model, feature_list_file: Optional[str] = None) -> List[str]:
    """
    Get features used by model either from model or feature list file.
    
    Args:
        model: PMML model
        feature_list_file: Path to feature list file (optional)
        
    Returns:
        List of feature names
    """
    if feature_list_file and os.path.exists(feature_list_file):
        return load_feature_list(feature_list_file)
    
    try:
        if hasattr(model, 'inputNames'):
            return model.inputNames
        elif hasattr(model, 'getInputNames'):
            return model.getInputNames()
        else:
            print("无法直接从模型获取特征列表，需要提供特征列表文件")
            return []
    except Exception as e:
        print(f"从模型提取特征列表失败: {str(e)}")
        return []

def prepare_prediction_data(df: pd.DataFrame, 
                           feature_cols: List[str], 
                           id_cols: List[str] = ID_COLS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for prediction.
    
    Args:
        df: Input DataFrame
        feature_cols: Feature columns to use
        id_cols: ID columns to keep
        
    Returns:
        Tuple of (features DataFrame, IDs DataFrame)
    """
    # Preprocess data
    df = preprocess_data(df, feature_cols, is_training=False)
    
    # Extract features and IDs
    X = df[feature_cols]
    ids_df = df[id_cols] if all(col in df.columns for col in id_cols) else None
    
    return X, ids_df

def predict_with_model(model: Model, 
                      X: pd.DataFrame, 
                      threshold: float = 0.5, 
                      output_col: str = 'score',
                      scale_factor: float = 1000) -> pd.DataFrame:
    """
    Make predictions with PMML model.
    
    Args:
        model: PMML model
        X: Features DataFrame
        threshold: Classification threshold
        output_col: Column name for prediction scores
        scale_factor: Factor to scale probabilities (e.g., 1000 for integer scores)
        
    Returns:
        DataFrame with predictions
    """
    # Get predictions
    predictions = model.predict(X)
    
    # Extract probability column
    probability_key = f'probability({1})'
    if probability_key in predictions.columns:
        probability_col = predictions[probability_key]
    else:
        # Try to find any probability column
        prob_cols = [col for col in predictions.columns if 'probability' in col]
        if prob_cols:
            probability_col = predictions[prob_cols[0]]
        else:
            raise ValueError("无法找到概率列，请检查模型输出")
    
    # Convert to proper Series/array if needed
    if isinstance(probability_col, pd.Series):
        probs = probability_col.values
    else:
        probs = probability_col
    
    # Calculate scores and predictions
    scores = probs * scale_factor
    preds = (probs >= threshold).astype(int)
    
    # Create results DataFrame
    results_df = pd.DataFrame()
    results_df[output_col] = scores
    results_df['prediction'] = preds
    results_df['probability'] = probs
    
    return results_df

def save_feature_list(features: List[str], output_file: str) -> None:
    """
    Save feature list to file.
    
    Args:
        features: List of feature names
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"特征列表已保存至 {output_file}") 