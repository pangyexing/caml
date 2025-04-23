#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the model pipeline.
Orchestrates data loading, preprocessing, modeling, and evaluation.
"""

import os
import sys
import argparse
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Import local modules
from src.config import ensure_dirs, DEFAULT_TARGET, RESULTS_DIR, MODEL_DIR
from src.preprocessing import (
    preprocess_data, 
    split_by_time,
    check_duplicate_features,
    check_duplicate_keys,
    merge_feature_files,
    process_sample_files
)
from src.feature_engineering import analyze_feature_stability
from src.modeling import two_stage_modeling_pipeline
from src.hyperopt_tuning import HyperparameterOptimizer
from src.deployment import deploy_model
from src.utils import (
    analyze_label_distribution,
    export_model_summary,
    plot_metrics_comparison,
    timer
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run machine learning pipeline for customer conversion prediction.')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'tune', 'deploy', 'check'],
                        help='Pipeline mode: train (default), tune, deploy, or check (data validation)')
    
    parser.add_argument('--data', type=str, default='merged_data.csv',
                        help='Path to merged data file (default: merged_data.csv)')
    
    parser.add_argument('--features', type=str, nargs='+', default=None,
                        help='Paths to feature files (for check or merge mode)')
    
    parser.add_argument('--sample1', type=str, default=None,
                        help='Path to first sample file (for merge mode)')
    
    parser.add_argument('--sample2', type=str, default=None,
                        help='Path to second sample file (for merge mode)')
    
    parser.add_argument('--target', type=str, default=DEFAULT_TARGET,
                        help=f'Target variable (default: {DEFAULT_TARGET})')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (for deploy mode)')
    
    parser.add_argument('--features-file', type=str, default=None,
                        help='Path to feature list file (for deploy mode)')
    
    parser.add_argument('--max-evals', type=int, default=50,
                        help='Maximum evaluations for hyperparameter tuning (default: 50)')
    
    return parser.parse_args()

@timer
def run_data_check(feature_files: List[str]):
    """
    Run data validation checks.
    
    Args:
        feature_files: List of feature file paths
    """
    print("\n=== Running data validation checks ===")
    
    # Check for duplicate features across files
    print("\nChecking for duplicate features across files...")
    check_duplicate_features(feature_files)
    
    # Check for duplicate keys within each file
    print("\nChecking for duplicate keys within each file...")
    check_duplicate_keys(feature_files)
    
    print("\nData validation checks completed.")

@timer
def run_data_merge(feature_files: List[str], sample_file1: str, sample_file2: str):
    """
    Merge feature files and process with sample files.
    
    Args:
        feature_files: List of feature file paths
        sample_file1: First sample file path
        sample_file2: Second sample file path
        
    Returns:
        Merged DataFrame or None if error
    """
    print("\n=== Running data merge ===")
    
    # Merge feature files
    print("\nMerging feature files...")
    merged_df = merge_feature_files(feature_files)
    
    if merged_df is None:
        print("Error: Failed to merge feature files.")
        return None
    
    print(f"Merged features dataset shape: {merged_df.shape}")
    
    # Process sample files if provided
    if sample_file1 and sample_file2:
        print("\nProcessing sample files...")
        result_df1, result_df2 = process_sample_files(merged_df, sample_file1, sample_file2)
        
        if result_df1 is not None:
            result_df1.to_csv("merged_sample1.csv", index=False)
            print(f"Saved merged sample1 dataset to merged_sample1.csv, shape: {result_df1.shape}")
        
        if result_df2 is not None:
            result_df2.to_csv("merged_sample2.csv", index=False)
            print(f"Saved merged sample2 dataset to merged_sample2.csv, shape: {result_df2.shape}")
        
        # Return the first processed DataFrame for further analysis
        return result_df1
    
    # Save merged dataset
    merged_df.to_csv("merged_features.csv", index=False)
    print(f"Saved merged features dataset to merged_features.csv")
    
    return merged_df

@timer
def run_training(data_file: str, target: str = DEFAULT_TARGET):
    """
    Run model training pipeline.
    
    Args:
        data_file: Data file path
        target: Target variable
        
    Returns:
        Training results or None if error
    """
    print(f"\n=== Running training pipeline for target: {target} ===")
    
    # Create output directories
    ensure_dirs()
    
    # Load data
    try:
        print(f"Loading data from {data_file}...")
        merged_df = pd.read_csv(data_file)
        print(f"Loaded data shape: {merged_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Analyze label distribution
    label_stats = analyze_label_distribution(merged_df, target=target)
    
    # Split data by time
    print("\nSplitting data by time...")
    train_df, test_df, split_date = split_by_time(merged_df, train_ratio=0.8)
    
    # Save split datasets for later reuse
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(RESULTS_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test_data.csv"), index=False)
    
    # Run two-stage modeling pipeline
    print("\nRunning two-stage modeling pipeline...")
    result = two_stage_modeling_pipeline(train_df, test_df, target=target)
    
    # Save training results
    print("\nSaving training results...")
    export_model_summary(result, os.path.join(RESULTS_DIR, f"{target}_training_summary.txt"))
    
    # Plot metrics comparison
    plot_metrics_comparison(
        result['initial_metrics'], 
        result['final_metrics'], 
        title=f"{target.capitalize()} Model Improvement", 
        output_file=os.path.join(RESULTS_DIR, f"{target}_metrics_comparison.png")
    )
    
    print(f"\nTraining pipeline completed successfully.")
    print(f"Model saved to: {result['model_file']}")
    print(f"Selected features saved to: {result['feature_file']}")
    
    return result

@timer
def run_hyperparameter_tuning(target: str = DEFAULT_TARGET, max_evals: int = 50):
    """
    Run hyperparameter tuning.
    
    Args:
        target: Target variable
        max_evals: Maximum evaluations
        
    Returns:
        Tuning results or None if error
    """
    print(f"\n=== Running hyperparameter tuning for target: {target} ===")
    
    # Create output directories
    dirs = ensure_dirs()
    
    # Load training and test data
    train_file = os.path.join(RESULTS_DIR, "train_data.csv")
    test_file = os.path.join(RESULTS_DIR, "test_data.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Training or test data not found. Please run training pipeline first.")
        return None
    
    # Load feature list from the two-stage modeling pipeline
    feature_file = os.path.join(MODEL_DIR, f"{target}_selected_features.txt")
    if not os.path.exists(feature_file):
        print(f"Error: Selected feature list not found at {feature_file}. Please run training pipeline first.")
        return None
    
    try:
        # Load selected features
        print(f"Loading selected features from {feature_file}...")
        with open(feature_file, 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(selected_features)} selected features")
        
        # Load training and test data
        print(f"Loading training data from {train_file}...")
        train_df = pd.read_csv(train_file)
        print(f"Loaded training data shape: {train_df.shape}")
        
        print(f"Loading test data from {test_file}...")
        test_df = pd.read_csv(test_file)
        print(f"Loaded test data shape: {test_df.shape}")
        
        # Filter data to only include selected features and required columns
        required_cols = ['input_key', 'recall_date', target]
        feature_cols = [col for col in selected_features if col in train_df.columns]
        cols_to_keep = list(set(required_cols + feature_cols))
        
        train_df = train_df[cols_to_keep]
        test_df = test_df[cols_to_keep]
        print(f"Filtered training data shape: {train_df.shape}")
        print(f"Filtered test data shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create hyperparameter optimizer
    optimizer = HyperparameterOptimizer(
        train_df=train_df,
        test_df=test_df,
        target=target,
        max_evals=max_evals,
        output_dir=dirs['tuning_dir']
    )
    
    # Run optimization
    print(f"\nStarting hyperparameter optimization with {max_evals} evaluations...")
    result = optimizer.optimize()
    
    print(f"\nHyperparameter tuning completed.")
    print(f"Tuned model saved to: {result['model_path']}")
    
    return result

@timer
def run_deployment(model_path: str, data_file: str, target: str = DEFAULT_TARGET, feature_list_file: str = None):
    """
    Run model deployment.
    
    Args:
        model_path: Model file path
        data_file: Data file path
        target: Target variable
        feature_list_file: Feature list file path
        
    Returns:
        Deployment results or None if error
    """
    print(f"\n=== Running model deployment for target: {target} ===")
    
    # Create output directories
    dirs = ensure_dirs()
    
    # Load data
    try:
        print(f"Loading data from {data_file}...")
        test_df = pd.read_csv(data_file)
        print(f"Loaded data shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Run deployment
    try:
        result = deploy_model(
            model_path=model_path,
            test_df=test_df,
            target=target,
            feature_list_file=feature_list_file,
            output_dir=dirs['deployment_dir']
        )
        
        print(f"\nModel deployment completed successfully.")
        print(f"Predictions saved to: {result['predictions_file']}")
        
        return result
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Run appropriate pipeline based on mode
        if args.mode == 'check':
            if not args.features:
                print("Error: Feature files must be provided in check mode.")
                sys.exit(1)
            
            run_data_check(args.features)
        
        elif args.mode == 'merge':
            if not args.features:
                print("Error: Feature files must be provided in merge mode.")
                sys.exit(1)
            
            run_data_merge(args.features, args.sample1, args.sample2)
        
        elif args.mode == 'train':
            if not args.data:
                print("Error: Data file must be provided in train mode.")
                sys.exit(1)
            
            run_training(args.data, args.target)
        
        elif args.mode == 'tune':
            run_hyperparameter_tuning(args.target, args.max_evals)
        
        elif args.mode == 'deploy':
            if not args.model:
                print("Error: Model file must be provided in deploy mode.")
                sys.exit(1)
            
            if not args.data:
                print("Error: Data file must be provided in deploy mode.")
                sys.exit(1)
            
            run_deployment(args.model, args.data, args.target, args.features_file)
        
        else:
            print(f"Error: Unknown mode '{args.mode}'.")
            sys.exit(1)
        
        # Report total time
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 