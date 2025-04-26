#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing and cleaning functionality.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import EXCLUDE_COLS


def preprocess_data(
    df: pd.DataFrame, 
    time_column: str = 'recall_date',
    target: str = 'label_apply',
    train_ratio: float = 0.7,
    test_ratio: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data and split into training and testing sets.
    
    Args:
        df: DataFrame to preprocess
        time_column: Time column name
        target: Target label
        train_ratio: Training set ratio
        test_ratio: Testing set ratio
        random_state: Random state for reproducibility
    
    Returns:
        train_df: Preprocessed training DataFrame
        test_df: Preprocessed testing DataFrame
    """
    print(f"开始数据预处理和划分，数据量: {len(df)}")
    
    # Convert date column to datetime
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d')
    
    # Fill missing values with appropriate strategies
    for col in df.columns:
        if col not in EXCLUDE_COLS:
            if df[col].dtype in ['float64', 'int64']:
                # For numeric columns, fill with median to minimize outlier impact
                df[col] = df[col].fillna(df[col].median())
            else:
                # For categorical columns, fill with the most common value
                most_common = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(most_common)
    
    # Drop any rows with missing target
    if target in df.columns:
        df = df.dropna(subset=[target])
    
    # Split data - Time-based or random based on presence of time column
    if time_column in df.columns and not df[time_column].isnull().all():
        print("基于时间列进行数据集划分")
        # Sort by time
        df = df.sort_values(by=time_column)
        # Split into train and test
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
    else:
        print("随机划分数据集")
        # Random split
        train_df = df.sample(frac=train_ratio, random_state=random_state)
        test_df = df.drop(train_df.index)
    
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # Check target distribution
    if target in train_df.columns:
        train_pos = train_df[target].sum()
        train_neg = len(train_df) - train_pos
        test_pos = test_df[target].sum()
        test_neg = len(test_df) - test_pos
        
        print(f"训练集目标分布: 正样本 {train_pos} ({train_pos/len(train_df):.2%}), 负样本 {train_neg} ({train_neg/len(train_df):.2%})")
        print(f"测试集目标分布: 正样本 {test_pos} ({test_pos/len(test_df):.2%}), 负样本 {test_neg} ({test_neg/len(test_df):.2%})")
    
    return train_df, test_df


def merge_feature_files(
    feature_files: List[str], 
    sample_file1: Optional[str] = None,
    sample_file2: Optional[str] = None, 
    key_column: str = 'input_key'
) -> pd.DataFrame:
    """
    Merge multiple feature files with sample files.
    
    Args:
        feature_files: List of feature file paths
        sample_file1: First sample file path
        sample_file2: Second sample file path
        key_column: Key column name for merging
    
    Returns:
        Merged DataFrame
    """
    feature_dfs = []
    
    # Load feature files
    for file in feature_files:
        print(f"加载特征文件: {file}")
        # Auto-detect file format
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.parquet'):
            df = pd.read_parquet(file)
        elif file.endswith('.pkl') or file.endswith('.pickle'):
            df = pd.read_pickle(file)
        else:
            raise ValueError(f"Unsupported file format: {file}")
        
        # Check if key column exists
        if key_column not in df.columns:
            raise ValueError(f"Key column '{key_column}' not found in {file}")
        
        feature_dfs.append(df)
    
    # Check if feature files exist
    if not feature_dfs:
        raise ValueError("No valid feature files provided")
    
    # Merge feature files
    merged_df = feature_dfs[0]
    for i, df in enumerate(feature_dfs[1:], 2):
        print(f"合并第 {i} 个特征文件，共 {len(feature_dfs)} 个")
        merged_df = pd.merge(merged_df, df, on=key_column, how='outer')
    
    # Merge with sample files if provided
    if sample_file1:
        print(f"合并样本文件 1: {sample_file1}")
        # Auto-detect file format
        if sample_file1.endswith('.csv'):
            sample_df1 = pd.read_csv(sample_file1)
        elif sample_file1.endswith('.parquet'):
            sample_df1 = pd.read_parquet(sample_file1)
        elif sample_file1.endswith('.pkl') or sample_file1.endswith('.pickle'):
            sample_df1 = pd.read_pickle(sample_file1)
        else:
            raise ValueError(f"Unsupported file format: {sample_file1}")
        
        # Check if key column exists
        if key_column not in sample_df1.columns:
            raise ValueError(f"Key column '{key_column}' not found in {sample_file1}")
        
        # Merge with sample file
        merged_df = pd.merge(merged_df, sample_df1, on=key_column, how='inner')
    
    if sample_file2:
        print(f"合并样本文件 2: {sample_file2}")
        # Auto-detect file format
        if sample_file2.endswith('.csv'):
            sample_df2 = pd.read_csv(sample_file2)
        elif sample_file2.endswith('.parquet'):
            sample_df2 = pd.read_parquet(sample_file2)
        elif sample_file2.endswith('.pkl') or sample_file2.endswith('.pickle'):
            sample_df2 = pd.read_pickle(sample_file2)
        else:
            raise ValueError(f"Unsupported file format: {sample_file2}")
        
        # Check if key column exists
        if key_column not in sample_df2.columns:
            raise ValueError(f"Key column '{key_column}' not found in {sample_file2}")
        
        # Merge with sample file
        merged_df = pd.merge(merged_df, sample_df2, on=key_column, how='inner')
    
    # Report on the merged dataset
    print(f"合并结果: {len(merged_df)} 行, {len(merged_df.columns)} 列")
    
    return merged_df


def check_feature_files(feature_files: List[str], key_column: str = 'input_key') -> Dict[str, Any]:
    """
    Check feature files for duplicate features and keys.
    
    Args:
        feature_files: List of feature file paths
        key_column: Key column name
    
    Returns:
        Dictionary with check results
    """
    results = {'status': 'success', 'issues': []}
    feature_maps = {}
    
    # Check each file
    for file in feature_files:
        print(f"检查文件: {file}")
        
        # Load file
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.parquet'):
            df = pd.read_parquet(file)
        elif file.endswith('.pkl') or file.endswith('.pickle'):
            df = pd.read_pickle(file)
        else:
            issue = f"不支持的文件格式: {file}"
            results['issues'].append(issue)
            continue
        
        # Check for key column
        if key_column not in df.columns:
            issue = f"键列 '{key_column}' 在文件 {file} 中不存在"
            results['issues'].append(issue)
            continue
        
        # Check for duplicate keys
        dup_keys = df[key_column].duplicated().sum()
        if dup_keys > 0:
            issue = f"文件 {file} 中存在 {dup_keys} 个重复键"
            results['issues'].append(issue)
        
        # Check feature names
        for col in df.columns:
            if col != key_column:
                if col in feature_maps:
                    issue = f"特征名称 '{col}' 在多个文件中重复: {file} 和 {feature_maps[col]}"
                    results['issues'].append(issue)
                else:
                    feature_maps[col] = file
    
    # Set status
    if results['issues']:
        results['status'] = 'issues_found'
        print(f"检查完成，发现 {len(results['issues'])} 个问题")
    else:
        print("检查完成，未发现问题")
    
    return results 