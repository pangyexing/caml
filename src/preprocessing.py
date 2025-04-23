#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing module for ML models.
Handles data cleaning, feature preparation, and dataset splitting.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

def preprocess_data(df: pd.DataFrame, 
                    feature_cols: List[str], 
                    is_training: bool = True,
                    keep_only_features: bool = False) -> pd.DataFrame:
    """
    Unified data preprocessing function for training and prediction.
    
    Args:
        df: Dataset to process
        feature_cols: Feature columns
        is_training: Whether this is training data (affects logging)
        keep_only_features: Whether to keep only feature columns and essential ID/label columns
    
    Returns:
        Processed dataset
    """
    # Check and remove rows where all features are zero
    zero_rows = (df[feature_cols] == 0).all(axis=1)
    
    if zero_rows.any():
        removed_count = zero_rows.sum()
        total_count = len(df)
        removed_percentage = (removed_count / total_count) * 100
        stage = "训练集" if is_training else "测试集"
        print(f"{stage}中移除 {removed_count} 行特征全为0的数据，占比 {removed_percentage:.2f}%")
        df = df[~zero_rows]
    
    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # Keep only necessary columns if specified
    if keep_only_features:
        # Keep essential columns: ID, date, labels
        necessary_cols = ['input_key', 'recall_date']
        label_cols = [col for col in df.columns if col.startswith('label_')]
        
        # Time bins and score columns
        additional_cols = []
        if 'time_bin' in df.columns:
            additional_cols.append('time_bin')
        if 'score' in df.columns:
            additional_cols.append('score')
            
        # Merge all columns to keep
        cols_to_keep = necessary_cols + label_cols + additional_cols + feature_cols
        # Only keep columns that exist in df
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[cols_to_keep]
    
    return df

def split_by_time(df: pd.DataFrame, 
                  train_ratio: float = 0.8, 
                  time_column: str = 'recall_date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split dataset into training and validation sets by time.
    
    Args:
        df: Dataset to split 
        train_ratio: Training set ratio
        time_column: Time column name
    
    Returns:
        train_df: Training dataset
        test_df: Testing dataset
        split_date: Split date
    """
    df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d')
    
    unique_dates = sorted(df[time_column].unique())
    
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]
    
    train_df = df[df[time_column] <= split_date]
    test_df = df[df[time_column] > split_date]
    
    print(f"数据集按时间拆分:")
    print(f"分割日期: {split_date}")
    print(f"训练集: {len(train_df)} 条记录 ({train_df[time_column].min()} 至 {train_df[time_column].max()})")
    print(f"验证集: {len(test_df)} 条记录 ({test_df[time_column].min()} 至 {test_df[time_column].max()})")
    
    return train_df, test_df, split_date

def check_duplicate_features(feature_files: List[str]) -> Dict[str, List[str]]:
    """
    Check for duplicate feature names across multiple feature files.
    
    Args:
        feature_files: List of feature file paths
    
    Returns:
        Dictionary of duplicate features and their source files
    """
    import os
    all_features = {}
    
    for file_path in feature_files:
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            feature_cols = [col for col in df.columns if col not in ['input_key', 'recall_date', 'customer_name']]
            
            for feature in feature_cols:
                if feature in all_features:
                    all_features[feature].append(file_name)
                else:
                    all_features[feature] = [file_name]
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    duplicate_features = {feature: files for feature, files in all_features.items() if len(files) > 1}
    if duplicate_features:
        print("发现重复的特征名:")
        for feature, files in duplicate_features.items():
            print(f"特征 '{feature}' 出现在文件: {', '.join(files)}")
    else:
        print("没有发现重复的特征名")
    
    return duplicate_features

def check_duplicate_keys(feature_files: List[str]) -> None:
    """
    Check for duplicate primary keys in feature files.
    
    Args:
        feature_files: List of feature file paths
    """
    import os
    for file_path in feature_files:
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            duplicates = df.duplicated(subset=['input_key', 'recall_date'], keep=False)
            if duplicates.any():
                print(f"文件 {file_name} 中发现重复的主键:")
                print(df[duplicates][['input_key', 'recall_date']])
            else:
                print(f"文件 {file_name} 中没有重复的主键")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

def merge_feature_files(feature_files: List[str]) -> Optional[pd.DataFrame]:
    """
    Merge multiple feature files into a single DataFrame.
    
    Args:
        feature_files: List of feature file paths
        
    Returns:
        Merged DataFrame or None if error
    """
    dfs = []
    
    for file_path in feature_files:
        try:
            df = pd.read_csv(file_path)
            if 'customer_name' in df.columns:
                df = df.drop(columns=['customer_name'])
            dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    if not dfs:
        return None
    
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on=['input_key', 'recall_date'], how='outer')
    
    return merged_df

def process_sample_files(merged_df: pd.DataFrame, 
                         sample_file1: str, 
                         sample_file2: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Process and merge sample files with feature data.
    
    Args:
        merged_df: Merged feature DataFrame 
        sample_file1: First sample file path
        sample_file2: Second sample file path
        
    Returns:
        Tuple of processed DataFrames (result_df1, result_df2)
    """
    if merged_df is None:
        print("没有有效的合并特征数据")
        return None, None
    
    merged_df = merged_df.fillna(0.0)
    
    try:
        sample1_df = pd.read_csv(sample_file1)
        original_sample1_count = len(sample1_df)
        if 'speaking_duration' in sample1_df.columns:
            sample1_df = sample1_df.drop(columns=['speaking_duration'])
        if 'label_intention' in sample1_df.columns:
            sample1_df = sample1_df.drop(columns=['label_intention'])
        
        result_df1 = pd.merge(merged_df, sample1_df, on=['input_key', 'recall_date'], how='inner')
        match_rate1 = len(result_df1) / original_sample1_count * 100
        print(f"Sample file 1 match rate: {match_rate1:.2f}% ({len(result_df1)}/{original_sample1_count} records matched)")
        
        sample2_df = pd.read_csv(sample_file2)
        original_sample2_count = len(sample2_df)
        
        if 'credit_level' in sample2_df.columns:
            sample2_df['credit_level'] = sample2_df['credit_level'].apply(lambda x: 1 if x in ['A', 'B'] else 0)
        
        result_df2 = pd.merge(merged_df, sample2_df, on=['input_key', 'recall_date'], how='inner')
        match_rate2 = len(result_df2) / original_sample2_count * 100
        print(f"Sample file 2 match rate: {match_rate2:.2f}% ({len(result_df2)}/{original_sample2_count} records matched)")
        
        return result_df1, result_df2
    
    except Exception as e:
        print(f"处理样本文件时出错: {e}")
        return None, None 