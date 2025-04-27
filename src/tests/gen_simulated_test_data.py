"""
label_file1 = "data/label_file1.csv"
format:
input_key,recall_date,label_register,label_apply,label_approve
recall_date: 20240101
label_register: 0 or 1
label_apply: 0 or 1
label_approve: 0 or 1

label_file2 = "data/label_file2.csv"
format:
input_key,recall_date,label_credit,credit_level
recall_date: 20240101
label_credit : 0 or 1
credit_level: 0 or 1

feature_file1 = "data/feature_file1.csv"
format:
input_key,recall_date,feature_1 ... feature_300
recall_date: 20240101
feature values: >=0.0 or NA

feature_file2 = "data/feature_file2.csv"
format:
input_key,recall_date,feature_301 ... feature_500
recall_date: 20240101
feature values: >=0.0 or NA

这四个文件的（input_key，recall_date）是大部分相同的，大约有85%的重叠率。

"""

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


def generate_test_data(output_dir: str = "data", num_samples: int = 10000, num_dates: int = 5, 
                      seed: int = 42, na_ratio: float = 0.05) -> None:
    """
    Generate simulated test data for the customer conversion model.
    
    Args:
        output_dir: Directory to save the generated files
        num_samples: Number of unique input_keys to generate
        num_dates: Number of different recall dates to use
        seed: Random seed for reproducibility
        na_ratio: Ratio of NA values to include in feature data (0.0 to 1.0)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate input keys
    input_keys = [f"user_{i:06d}" for i in range(num_samples)]
    
    # Generate recall dates (recent dates)
    base_date = datetime(2024, 1, 1)
    recall_dates = [(base_date - timedelta(days=i*30)).strftime("%Y%m%d") 
                   for i in range(num_dates)]
    
    # Create combinations of input_keys and recall_dates
    # Not all users will have all dates
    all_combinations = []
    for key in input_keys:
        # Each user has between 1 and num_dates recall dates
        num_user_dates = random.randint(1, num_dates)
        user_dates = random.sample(recall_dates, num_user_dates)
        for date in user_dates:
            all_combinations.append((key, date))
    
    # Shuffle combinations
    random.shuffle(all_combinations)
    
    # Generate label_file1
    label_df1 = pd.DataFrame(all_combinations, columns=['input_key', 'recall_date'])
    
    # Generate labels with realistic conversion rates
    # Typically: register > apply > approve
    label_df1['label_register'] = np.random.binomial(1, 0.3, size=len(label_df1))
    
    # Apply only if registered
    label_df1['label_apply'] = 0
    label_df1.loc[label_df1['label_register'] == 1, 'label_apply'] = \
        np.random.binomial(1, 0.4, size=label_df1['label_register'].sum())
    
    # Approve only if applied
    label_df1['label_approve'] = 0
    label_df1.loc[label_df1['label_apply'] == 1, 'label_approve'] = \
        np.random.binomial(1, 0.6, size=label_df1['label_apply'].sum())
    
    # Generate label_file2 (subset of users from label_file1)
    subset_indices = np.random.choice(
        len(label_df1), 
        size=int(len(label_df1) * 0.8), 
        replace=False
    )
    label_df2 = label_df1.iloc[subset_indices][['input_key', 'recall_date']].copy()
    
    # Generate credit labels
    label_df2['label_credit'] = np.random.binomial(1, 0.25, size=len(label_df2))
    
    # Credit level only if credit label is 1
    label_df2['credit_level'] = 0
    label_df2.loc[label_df2['label_credit'] == 1, 'credit_level'] = \
        np.random.choice([1, 2, 3], size=label_df2['label_credit'].sum())
    
    # Generate feature files
    # Feature file 1: features 1-300
    feature_df1 = label_df1[['input_key', 'recall_date']].copy()
    
    # Generate features with some correlation to labels
    # Create a dictionary to store feature data before adding to DataFrame
    feature_data1 = {}
    
    # Strongly predictive features (1-50)
    for i in range(1, 51):
        base = label_df1['label_register'] * 2 + np.random.normal(0, 0.5, size=len(label_df1))
        feature_data1[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.5, size=len(label_df1)))
    
    # Moderately predictive features (51-100)
    for i in range(51, 101):
        base = label_df1['label_apply'] * 1.5 + np.random.normal(0, 0.7, size=len(label_df1))
        feature_data1[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.5, size=len(label_df1)))
    
    # Weakly predictive features (101-150)
    for i in range(101, 151):
        base = label_df1['label_approve'] * 1.2 + np.random.normal(0, 1.0, size=len(label_df1))
        feature_data1[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.5, size=len(label_df1)))
    
    # Noise features (151-300)
    for i in range(151, 301):
        base = np.random.normal(0, 1.0, size=len(label_df1))
        feature_data1[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.5, size=len(label_df1)))
    
    # Add all features to the DataFrame at once using pd.concat
    feature_df1 = pd.concat([feature_df1, pd.DataFrame(feature_data1)], axis=1)
    
    # Feature file 2: features 301-500
    # Use 80% of the same input_keys as feature_file1 but with some different dates
    subset_indices = np.random.choice(
        len(feature_df1), 
        size=int(len(feature_df1) * 0.9), 
        replace=False
    )
    feature_df2 = feature_df1.iloc[subset_indices][['input_key', 'recall_date']].copy()
    
    # Add some new combinations
    new_combinations = []
    for key in random.sample(input_keys, int(num_samples * 0.1)):
        date = random.choice(recall_dates)
        new_combinations.append((key, date))
    
    new_df = pd.DataFrame(new_combinations, columns=['input_key', 'recall_date'])
    feature_df2 = pd.concat([feature_df2, new_df], ignore_index=True)
    
    # Generate features 301-500
    feature_data2 = {}
    
    # Some predictive features (301-350)
    for i in range(301, 351):
        base = np.random.normal(1.0, 1.0, size=len(feature_df2))
        feature_data2[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.3, size=len(feature_df2)))
    
    # Mostly noise features (351-500)
    for i in range(351, 501):
        base = np.random.normal(0.5, 1.5, size=len(feature_df2))
        feature_data2[f'feature_{i}'] = np.maximum(0, base + np.random.exponential(0.3, size=len(feature_df2)))
    
    # Add all features to the DataFrame at once
    feature_df2 = pd.concat([feature_df2, pd.DataFrame(feature_data2)], axis=1)
    
    # Add NA values to features
    if na_ratio > 0:
        # For feature_df1
        feature_cols1 = [col for col in feature_df1.columns if col.startswith('feature_')]
        for col in feature_cols1:
            # Create a mask for inserting NA values
            na_mask = np.random.random(len(feature_df1)) < na_ratio
            feature_df1.loc[na_mask, col] = np.nan

        # For feature_df2
        feature_cols2 = [col for col in feature_df2.columns if col.startswith('feature_')]
        for col in feature_cols2:
            # Create a mask for inserting NA values
            na_mask = np.random.random(len(feature_df2)) < na_ratio
            feature_df2.loc[na_mask, col] = np.nan
    
    # Save files
    label_df1.to_csv(os.path.join(output_dir, 'label_file1.csv'), index=False)
    label_df2.to_csv(os.path.join(output_dir, 'label_file2.csv'), index=False)
    feature_df1.to_csv(os.path.join(output_dir, 'feature_file1.csv'), index=False)
    feature_df2.to_csv(os.path.join(output_dir, 'feature_file2.csv'), index=False)
    
    # Calculate NA statistics
    na_count1 = feature_df1[feature_cols1].isna().sum().sum()
    na_count2 = feature_df2[feature_cols2].isna().sum().sum()
    total_cells1 = len(feature_df1) * len(feature_cols1)
    total_cells2 = len(feature_df2) * len(feature_cols2)
    actual_na_ratio1 = na_count1 / total_cells1
    actual_na_ratio2 = na_count2 / total_cells2
    
    print(f"Generated test data in {output_dir}:")
    print(f"  label_file1.csv: {len(label_df1)} rows")
    print(f"  label_file2.csv: {len(label_df2)} rows")
    print(f"  feature_file1.csv: {len(feature_df1)} rows, {len(feature_df1.columns)-2} features")
    print(f"    NA ratio: {actual_na_ratio1:.2%} ({na_count1} cells)")
    print(f"  feature_file2.csv: {len(feature_df2)} rows, {len(feature_df2.columns)-2} features")
    print(f"    NA ratio: {actual_na_ratio2:.2%} ({na_count2} cells)")

if __name__ == "__main__":
    generate_test_data()
