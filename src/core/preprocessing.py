#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing and cleaning functionality.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.core.config import EXCLUDE_COLS, ID_COLS


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
                df[col] = df[col].fillna(0.0)
    
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
    sample_file: str = None,
    key_column: str = 'input_key'
) -> pd.DataFrame:
    """
    Merge multiple feature files with sample files.
    
    Args:
        feature_files: List of feature file paths
        sample_file: Sample file path
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
        
        # 只保留ID_COLS和EXCLUDE_COLS之外的特征列
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
        keep_cols = ID_COLS + feature_cols
        df = df[keep_cols]
        print(f"处理后保留的列数: {len(keep_cols)}")
        
        feature_dfs.append(df)
    
    # Check if feature files exist
    if not feature_dfs:
        raise ValueError("No valid feature files provided")
    
    # Merge feature files
    merged_df = feature_dfs[0]
    for i, df in enumerate(feature_dfs[1:], 2):
        print(f"合并第 {i} 个特征文件，共 {len(feature_dfs)} 个")
        # 使用ID_COLS中的所有列作为合并键，如果存在于两个DataFrame中
        merge_on = [col for col in ID_COLS if col in merged_df.columns and col in df.columns]
        if not merge_on:
            # 如果没有共同的ID列，则使用默认的key_column
            merge_on = key_column
        
        # 为重复列名添加后缀
        merged_df = pd.merge(
            merged_df, 
            df, 
            on=merge_on, 
            how='outer',
            suffixes=('', f'_{i}')  # 使用数字后缀来避免列名冲突
        )
    
    # Merge with sample files if provided
    if sample_file:
        print(f"合并样本文件: {sample_file}")
        # Auto-detect file format
        if sample_file.endswith('.csv'):
            sample_df = pd.read_csv(sample_file)
        elif sample_file.endswith('.parquet'):
            sample_df = pd.read_parquet(sample_file)
        elif sample_file.endswith('.pkl') or sample_file.endswith('.pickle'):
            sample_df = pd.read_pickle(sample_file)
        else:
            raise ValueError(f"Unsupported file format: {sample_file}")
        
        # Check if key column exists
        if key_column not in sample_df.columns:
            raise ValueError(f"Key column '{key_column}' not found in {sample_file}")
        
        # 使用ID_COLS中的所有列作为合并键，如果存在于两个DataFrame中
        merge_on = [col for col in ID_COLS if col in merged_df.columns and col in sample_df.columns]
        if not merge_on:
            # 如果没有共同的ID列，则使用默认的key_column
            merge_on = key_column
            
        # Merge with sample file
        merged_df = pd.merge(
            merged_df, 
            sample_df, 
            on=merge_on, 
            how='inner',
            suffixes=('', '_sample')
        )
    
    # Report on the merged dataset
    print(f"合并结果: {len(merged_df)} 行, {len(merged_df.columns)} 列")
    
    return merged_df


def check_feature_files(feature_files: List[str], key_columns: list = ID_COLS) -> Dict[str, Any]:
    """
    Check feature files for duplicate features and keys.
    
    Args:
        feature_files: List of feature file paths
        key_columns: List of key column names that together form the primary key
    
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
        
        # 确定要检查的主键列
        id_cols_present = [col for col in key_columns if col in df.columns]
        
        if not id_cols_present:
            # 如果没有任何配置的主键列存在，报告错误
            missing_cols = ', '.join(key_columns)
            issue = f"文件 {file} 中不存在任何配置的主键列: {missing_cols}"
            results['issues'].append(issue)
            continue
        
        # 检查组合主键是否有重复
        if len(id_cols_present) > 0:
            # 检查所有主键列组合是否有重复
            dup_keys = df.duplicated(subset=id_cols_present).sum()
            if dup_keys > 0:
                key_desc = '、'.join(id_cols_present) if len(id_cols_present) > 1 else id_cols_present[0]
                issue = f"文件 {file} 中组合主键 '{key_desc}' 存在 {dup_keys} 个重复记录"
                results['issues'].append(issue)
        
        # Check feature names
        for col in df.columns:
            if col not in id_cols_present:  # 排除所有主键列
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