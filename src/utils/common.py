#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common utility functions.
"""

import functools
import os
import time
from typing import Any, Callable, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import json


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Started {func.__name__} at {time.strftime('%H:%M:%S')}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Finished {func.__name__} in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        return result
    return wrapper


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get memory usage statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    # Get memory usage by data type
    dtype_memory = {}
    for dtype in df.dtypes.unique():
        dtype_cols = df.select_dtypes(include=[dtype]).columns
        dtype_memory[str(dtype)] = memory_usage[dtype_cols].sum()
    
    return {
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / (1024 * 1024),
        'memory_per_row_kb': total_memory / len(df) / 1024 if len(df) > 0 else 0,
        'memory_by_dtype': {k: v / (1024 * 1024) for k, v in dtype_memory.items()}
    }


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types and converting objects to categories.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame
    """
    result = df.copy()
    
    # Process each column
    for col in result.columns:
        # Skip special columns
        if col in ['input_key', 'recall_date']:
            continue
            
        col_type = result[col].dtype
        
        # Optimize integers
        if pd.api.types.is_integer_dtype(col_type):
            c_min = result[col].min()
            c_max = result[col].max()
            
            # Select appropriate integer type
            if c_min >= 0:
                if c_max < 2**8:
                    result[col] = result[col].astype(np.uint8)
                elif c_max < 2**16:
                    result[col] = result[col].astype(np.uint16)
                elif c_max < 2**32:
                    result[col] = result[col].astype(np.uint32)
            else:
                if c_min > -2**7 and c_max < 2**7:
                    result[col] = result[col].astype(np.int8)
                elif c_min > -2**15 and c_max < 2**15:
                    result[col] = result[col].astype(np.int16)
                elif c_min > -2**31 and c_max < 2**31:
                    result[col] = result[col].astype(np.int32)
        
        # Optimize floats
        elif pd.api.types.is_float_dtype(col_type):
            result[col] = pd.to_numeric(result[col], downcast='float')
            
        # Convert object to category if cardinality is low
        elif pd.api.types.is_object_dtype(col_type):
            if result[col].nunique() < len(result) * 0.5:  # If less than 50% unique values
                result[col] = result[col].astype('category')
    
    return result


def check_nulls(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Check for null values in DataFrame.
    
    Args:
        df: Input DataFrame
        threshold: Threshold for reporting high null rate columns
        
    Returns:
        Dictionary with null value statistics
    """
    # Calculate null counts and percentages
    null_counts = df.isnull().sum()
    null_pcts = null_counts / len(df)
    
    # Get columns with nulls
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    # Get columns with high null rate
    high_null_cols = null_pcts[null_pcts > threshold].index.tolist()
    
    return {
        'total_null_count': null_counts.sum(),
        'columns_with_nulls': len(cols_with_nulls),
        'column_null_counts': {col: null_counts[col] for col in cols_with_nulls},
        'column_null_pcts': {col: null_pcts[col] for col in cols_with_nulls},
        'high_null_rate_columns': high_null_cols
    }


def analyze_label_distribution(df: pd.DataFrame, target: str = 'label_apply') -> Dict[str, Any]:
    """
    Analyze the distribution of label/target values.
    
    Args:
        df: Input DataFrame
        target: Target column name
        
    Returns:
        Dictionary with label distribution statistics
    """
    if target not in df.columns:
        return {'status': 'error', 'message': f'Target column {target} not found'}
    
    # Calculate basic statistics
    label_counts = df[target].value_counts().to_dict()
    label_pcts = df[target].value_counts(normalize=True).to_dict()
    
    # For binary classification, get positive class statistics
    if len(label_counts) == 2:
        positive_count = label_counts.get(1, 0)
        negative_count = label_counts.get(0, 0)
        positive_pct = positive_count / (positive_count + negative_count) if positive_count + negative_count > 0 else 0
        
        # Return binary classification statistics
        return {
            'status': 'success',
            'type': 'binary',
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_pct': positive_pct,
            'class_imbalance_ratio': negative_count / positive_count if positive_count > 0 else float('inf')
        }
    else:
        # Return multi-class classification statistics
        return {
            'status': 'success',
            'type': 'multi-class',
            'class_counts': label_counts,
            'class_pcts': label_pcts,
            'num_classes': len(label_counts)
        }


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON serializable format, handling NumPy arrays, pandas DataFrames, and other
    non-serializable types.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, (datetime, np.ndarray)):
        return str(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy types.
    """
    def default(self, obj):
        return convert_to_serializable(obj)


def serialize_to_json(obj: Any, file_path: str, indent: int = 2) -> None:
    """
    Serialize an object to a JSON file, handling non-serializable types.
    
    Args:
        obj: The object to serialize
        file_path: Path to output file
        indent: JSON indentation level
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable format and save
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=indent, cls=NumpyEncoder)


def save_job_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save job results to file.
    
    Args:
        results: Dictionary of job results
        filename: Output filename
    """
    # Use the serialize_to_json utility function
    serialize_to_json(results, filename)


def load_job_results(filename: str) -> Dict[str, Any]:
    """
    Load job results from file.
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary of job results
    """
    import json
    
    with open(filename, 'r') as f:
        return json.load(f) 