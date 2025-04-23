#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter optimization module using Hyperopt.
Optimizes model parameters with focus on positive sample prediction.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Hyperopt imports
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope

# Sklearn and XGBoost imports
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from xgboost import XGBClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

# Import local modules
from src.preprocessing import preprocess_data
from src.config import TUNING_DIR

# Ignore warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """Hyperparameter optimization class using Hyperopt."""
    
    def __init__(self, 
                 train_df: pd.DataFrame, 
                 test_df: pd.DataFrame, 
                 target: str = 'label_apply', 
                 max_evals: int = 50,
                 output_dir: str = TUNING_DIR):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            target: Target variable name
            max_evals: Maximum evaluations for hyperopt
            output_dir: Output directory for results
        """
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.max_evals = max_evals
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare features and data
        self.feature_cols, self.X_train, self.y_train = self._prepare_features_labels(self.train_df)
        _, self.X_test, self.y_test = self._prepare_features_labels(self.test_df)
        
        # Calculate class balance for scale_pos_weight
        self.pos_count = self.y_train.sum()
        self.neg_count = len(self.y_train) - self.pos_count
        self.pos_weight = self.neg_count / self.pos_count
        
        # Define hyperparameter space
        self.space = self._define_parameter_space()
        
        # Results
        self.best_params = None
        self.best_model = None
        self.best_metrics = None
        self.trials = None
    
    def _prepare_features_labels(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_columns, X_features, y_target)
        """
        # Exclude non-feature columns
        exclude_cols = ['input_key', 'recall_date', 'label_register', 'label_apply', 'label_approve', 'time_bin', 'score']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Preprocess data
        df = preprocess_data(df, feature_cols)
        
        # Extract features and target
        X = df[feature_cols]
        y = df[self.target]
        
        return feature_cols, X, y
    
    def _define_parameter_space(self) -> Dict:
        """
        Define hyperparameter search space for XGBoost.
        
        Returns:
            Dictionary with hyperparameter space
        """
        # Define parameter space with focus on positive sample prediction
        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
            'gamma': hp.uniform('gamma', 0, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-4), np.log(1)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-4), np.log(10)),
            'scale_pos_weight': hp.quniform('scale_pos_weight', self.pos_weight*0.5, self.pos_weight*1.5, 0.1)
        }
        
        return space
    
    def objective(self, params: Dict) -> Dict:
        """
        Objective function for hyperparameter optimization.
        Optimizes for PR-AUC which focuses on positive sample prediction.
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            Evaluation results dictionary
        """
        start_time = time.time()
        
        # Ensure certain parameters are integers
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])
        
        # Create XGBoost model
        model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            seed=42,
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            scale_pos_weight=params['scale_pos_weight']
        )
        
        # Train model
        model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_test, self.y_test)],
            eval_metric='aucpr',  # Focus on PR-AUC
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predict
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate metrics at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_metrics = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            true_pos = np.sum((self.y_test == 1) & (y_pred == 1))
            false_pos = np.sum((self.y_test == 0) & (y_pred == 1))
            false_neg = np.sum((self.y_test == 1) & (y_pred == 0))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
            
            threshold_metrics.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2
            })
        
        elapsed_time = time.time() - start_time
        
        # Optimization target (negative PR-AUC for minimization)
        main_metric = -pr_auc
        
        return {
            'loss': main_metric,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'threshold_metrics': threshold_metrics,
            'status': STATUS_OK,
            'model': model,
            'elapsed_time': elapsed_time
        }
    
    def optimize(self) -> Dict:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with optimization results
        """
        print("="*80)
        print("开始超参数优化，专注提高正样本预测能力")
        print("="*80)
        
        print(f"训练集正负样本比例: 1:{self.neg_count/self.pos_count:.2f}")
        
        # Create trials object to store results
        self.trials = Trials()
        
        # Run optimization
        print(f"开始参数搜索，最大评估次数: {self.max_evals}")
        start_time = time.time()
        
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials
        )
        
        # Get best parameters
        self.best_params = space_eval(self.space, best)
        
        # Ensure integer parameters
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['min_child_weight'] = int(self.best_params['min_child_weight'])
        self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
        
        # Find best trial results
        best_trial = sorted(self.trials.results, key=lambda x: x['loss'])[0]
        self.best_model = best_trial['model']
        self.best_metrics = {
            'roc_auc': best_trial['roc_auc'],
            'pr_auc': best_trial['pr_auc'],
            'threshold_metrics': best_trial['threshold_metrics']
        }
        
        elapsed_time = time.time() - start_time
        print(f"参数搜索完成，耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        
        # Print best parameters
        print("\n最佳参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Print best performance
        print("\n最佳性能:")
        print(f"  ROC-AUC: {self.best_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {self.best_metrics['pr_auc']:.4f}")
        
        # Print metrics at different thresholds
        print("\n不同阈值下的性能:")
        for metrics in self.best_metrics['threshold_metrics']:
            thresh = metrics['threshold']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            f2 = metrics['f2']
            print(f"  阈值={thresh:.1f}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, F2={f2:.4f}")
        
        # Create final model with best parameters
        self._create_final_model()
        
        # Plot optimization history
        self.plot_optimization_history()
        
        return {
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'model_path': os.path.join(self.output_dir, f"{self.target}_tuned_model.pmml"),
            'feature_file': os.path.join(self.output_dir, f"{self.target}_features.txt"),
            'trials': self.trials
        }
    
    def _create_final_model(self) -> None:
        """
        Create and save final model using best parameters.
        """
        print("\n使用最佳参数重新训练最终模型...")
        final_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            seed=42,
            **self.best_params
        )
        
        # Create PMMLPipeline
        pipeline = PMMLPipeline([("classifier", final_model)])
        
        # Train model
        pipeline.fit(
            self.X_train, 
            self.y_train, 
            classifier__eval_set=[(self.X_test, self.y_test)],
            classifier__eval_metric='aucpr',
            classifier__early_stopping_rounds=50,
            classifier__verbose=False
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{self.target}_tuned_model.pmml")
        sklearn2pmml(pipeline, model_path, with_repr=True)
        
        # Save feature list
        feature_file = os.path.join(self.output_dir, f"{self.target}_features.txt")
        with open(feature_file, 'w') as f:
            for feature in self.feature_cols:
                f.write(f"{feature}\n")
        
        # Save tuning results
        results_file = os.path.join(self.output_dir, f"{self.target}_tuning_results.txt")
        with open(results_file, 'w') as f:
            f.write("超参数优化结果 - 正样本预测能力优化\n")
            f.write("="*50 + "\n\n")
            
            f.write("最佳参数:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n最佳性能:\n")
            f.write(f"  ROC-AUC: {self.best_metrics['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {self.best_metrics['pr_auc']:.4f}\n")
            
            f.write("\n不同阈值下的性能:\n")
            for metrics in self.best_metrics['threshold_metrics']:
                thresh = metrics['threshold']
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1']
                f2 = metrics['f2']
                f.write(f"  阈值={thresh:.1f}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, F2={f2:.4f}\n")
        
        print(f"\n优化结果已保存至: {results_file}")
        print(f"最终模型已保存至: {model_path}")
        print(f"特征列表已保存至: {feature_file}")
    
    def plot_optimization_history(self) -> None:
        """
        Plot optimization history and parameter importance.
        """
        print("\n绘制优化历史图表...")
        
        # Extract PR-AUC values from each iteration
        iterations = list(range(1, len(self.trials.results) + 1))
        pr_aucs = [-result['loss'] for result in self.trials.results]  # Convert back to positive
        
        # Plot PR-AUC vs. iteration
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, pr_aucs, 'b-', marker='o')
        plt.xlabel('迭代次数')
        plt.ylabel('PR-AUC')
        plt.title('PR-AUC随参数优化迭代变化')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimization_history.png'))
        plt.close()
        
        # Plot parameter importance: analyze relationship between each parameter and PR-AUC
        param_names = list(self.trials.trials[0]['misc']['vals'].keys())
        
        # Create scatter plots for each parameter
        plt.figure(figsize=(15, 10))
        n_params = len(param_names)
        rows = (n_params + 1) // 2
        for i, param in enumerate(param_names):
            plt.subplot(rows, 2, i + 1)
            
            param_values = []
            for trial in self.trials.trials:
                # Extract parameter value
                if param in trial['misc']['vals'] and len(trial['misc']['vals'][param]) > 0:
                    param_values.append(trial['misc']['vals'][param][0])
                else:
                    param_values.append(None)
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(param_values) if v is not None]
            valid_values = [param_values[i] for i in valid_indices]
            valid_pr_aucs = [pr_aucs[i] for i in valid_indices]
            
            plt.scatter(valid_values, valid_pr_aucs, alpha=0.6)
            plt.xlabel(param)
            plt.ylabel('PR-AUC')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'))
        plt.close()

def load_data(data_file: str) -> Optional[pd.DataFrame]:
    """
    Load dataset from file.
    
    Args:
        data_file: Data file path
        
    Returns:
        Loaded DataFrame or None if error
    """
    try:
        df = pd.read_csv(data_file)
        print(f"成功加载数据: {data_file}, 记录数: {len(df)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def main():
    """Main function for hyperparameter optimization."""
    start_time = time.time()
    
    try:
        # Set output directory
        output_dir = TUNING_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Load training and test data
        train_file = "optimization_results/train_data.csv"
        test_file = "optimization_results/test_data.csv"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print("训练集或测试集文件不存在，请先运行optimize_model.py")
            return
        
        train_df = load_data(train_file)
        test_df = load_data(test_file)
        
        if train_df is None or test_df is None:
            print("加载数据失败，请检查数据文件")
            return
        
        # Run hyperparameter optimization
        max_evals = 50  # Adjust evaluation count
        optimizer = HyperparameterOptimizer(
            train_df=train_df,
            test_df=test_df,
            target='label_apply',
            max_evals=max_evals,
            output_dir=output_dir
        )
        
        # Optimize
        results = optimizer.optimize()
        
        elapsed_time = time.time() - start_time
        print(f"\n全部完成，总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 