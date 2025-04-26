#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main executable script for customer conversion model.
"""

import argparse
import os
import sys

import pandas as pd

from src.core.preprocessing import (
    check_feature_files,
    merge_feature_files,
    preprocess_data,
)
from src.models.deployment import batch_prediction, deploy_model, load_feature_list
from src.models.hyperopt_tuning import hyperopt_xgb, plot_optimization_results
from src.models.training import two_stage_modeling_pipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Customer Conversion Model')
    
    # Mode argument
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['check', 'merge', 'train', 'tune', 'deploy'],
                        help='Operation mode')
    
    # Feature files arguments
    parser.add_argument('--features', type=str, nargs='+',
                        help='Feature files to process')
    
    # Sample files arguments
    parser.add_argument('--sample', type=str,
                        help='sample file')
    
    # Data file arguments
    parser.add_argument('--data', type=str,
                        help='Data file for training or deployment')
    
    # Target column
    parser.add_argument('--target', type=str, default='label_apply',
                        help='Target column name')
    
    # Resume from stage
    parser.add_argument('--resume-from', type=str,
                        choices=['start', 'initial_model', 'feature_analysis', 
                                'feature_selection', 'final_model'],
                        help='Resume training from a specific stage')
    
    # Hyperparameter tuning
    parser.add_argument('--max-evals', type=int, default=50,
                        help='Maximum number of hyperparameter evaluations')
    
    # Deployment
    parser.add_argument('--model', type=str,
                        help='Model file for deployment')
    
    # Batch prediction
    parser.add_argument('--key-column', type=str, default='input_key',
                        help='Key column for batch prediction')
    parser.add_argument('--output', type=str,
                        help='Output file for batch prediction')
    parser.add_argument('--features-file', type=str,
                        help='File containing list of features')
    parser.add_argument('--threshold', type=float,
                        help='Classification threshold')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Mode: check feature files for duplicates
    if args.mode == 'check':
        if not args.features:
            print("Error: --features argument is required for check mode")
            sys.exit(1)
        
        print(f"Checking {len(args.features)} feature files for duplicates...")
        results = check_feature_files(args.features)
        
        if results['status'] == 'success':
            print("No issues found")
        else:
            print(f"Found {len(results['issues'])} issues:")
            for issue in results['issues']:
                print(f"  - {issue}")
    
    # Mode: merge feature files
    elif args.mode == 'merge':
        if not args.features:
            print("Error: --features argument is required for merge mode")
            sys.exit(1)
        
        print(f"Merging {len(args.features)} feature files...")
        merged_df = merge_feature_files(
            args.features,
            args.sample
        )
        
        # Save merged dataset
        output_file = 'merged_features.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged dataset to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    # Mode: train model
    elif args.mode == 'train':
        if not args.data:
            print("Error: --data argument is required for train mode")
            sys.exit(1)
        
        print(f"Loading data from {args.data}...")
        
        # Load data
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            print(f"Error: Unsupported file format: {args.data}")
            sys.exit(1)
        
        print(f"Data loaded, shape: {df.shape}")
        
        # Preprocess and split data
        print("Preprocessing data...")
        train_df, test_df = preprocess_data(
            df, 
            target=args.target
        )
        
        # Run two-stage modeling pipeline
        results = two_stage_modeling_pipeline(
            train_df, 
            test_df, 
            target=args.target,
            resume_from=args.resume_from
        )
        
        print("Training completed")
    
    # Mode: hyperparameter tuning
    elif args.mode == 'tune':
        if not args.target:
            print("Error: --target argument is required for tune mode")
            sys.exit(1)
        
        # Check if we need to load data or we already have models
        features_file = "funnel_models/selected_features.txt"
        
        if not os.path.exists(features_file):
            print("Error: Feature list not found. Please run train mode first.")
            sys.exit(1)
        
        if args.data:
            print(f"Loading data from {args.data}...")
            
            # Load data
            if args.data.endswith('.csv'):
                df = pd.read_csv(args.data)
            elif args.data.endswith('.parquet'):
                df = pd.read_parquet(args.data)
            else:
                print(f"Error: Unsupported file format: {args.data}")
                sys.exit(1)
            
            print(f"Data loaded, shape: {df.shape}")
            
            # Preprocess and split data
            print("Preprocessing data...")
            train_df, test_df = preprocess_data(
                df, 
                target=args.target
            )
        else:
            print("Loading data from funnel_models directory...")
            
            # Try to load from saved files
            train_file = "funnel_models/train.csv"
            test_file = "funnel_models/test.csv"
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print("Error: Training data not found. Please use --data argument.")
                sys.exit(1)
            
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            print(f"Data loaded, train shape: {train_df.shape}, test shape: {test_df.shape}")
        
        # Load feature list
        features = load_feature_list(features_file)
        print(f"Loaded {len(features)} features from {features_file}")
        
        # Run hyperparameter optimization
        results = hyperopt_xgb(
            train_df,
            test_df,
            features,
            target=args.target,
            max_evals=args.max_evals
        )
        
        # Create visualization plots
        results_file = f"optimization_results/{args.target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_results.json"
        plot_optimization_results(results_file)
        
        print("Hyperparameter tuning completed")
    
    # Mode: deploy model
    elif args.mode == 'deploy':
        if not args.model:
            print("Error: --model argument is required for deploy mode")
            sys.exit(1)
        
        if not args.data:
            print("Error: --data argument is required for deploy mode")
            sys.exit(1)
        
        print(f"Loading data from {args.data}...")
        
        # Load data
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            print(f"Error: Unsupported file format: {args.data}")
            sys.exit(1)
        
        print(f"Data loaded, shape: {df.shape}")
        
        # Deploy model
        if args.target and args.target in df.columns:
            print(f"Deploying model with evaluation against {args.target}...")
            
            results = deploy_model(
                args.model,
                df,
                target=args.target,
                features_file=args.features_file,
                threshold=args.threshold
            )
        else:
            print("Deploying model for batch prediction...")
            
            if not args.features_file:
                print("Error: --features-file is required for batch prediction")
                sys.exit(1)
            
            if not args.output:
                args.output = 'predictions.csv'
            
            results = batch_prediction(
                args.model,
                df,
                key_column=args.key_column,
                features_file=args.features_file,
                output_file=args.output,
                threshold=args.threshold or 0.5
            )
        
        print("Deployment completed")
    
    else:
        print(f"Error: Unsupported mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main() 