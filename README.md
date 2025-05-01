# Customer Conversion Prediction Model

This repository contains a machine learning system specifically designed for predicting customer conversion events, with a focus on optimizing for positive sample prediction. The system includes a full pipeline from data preprocessing to hyperparameter tuning and model deployment.

## Features

- **Two-stage modeling approach**: Initial model for feature selection followed by optimized model for prediction
- **Focus on positive sample prediction**: Specifically optimized for PR-AUC metrics and positive sample recall
- **Feature importance analysis**: Multiple methods for understanding feature contributions (XGBoost importance, SHAP values)
- **Feature stability analysis**: Population Stability Index (PSI) to detect feature drift over time
- **Comprehensive hyperparameter optimization**: Using Hyperopt for finding optimal model parameters
- **Model deployment utilities**: Easy deployment with consistent feature preprocessing
- **Multi-threaded feature analysis**: Parallel processing for efficient feature evaluation
- **Visualization tools**: Comprehensive plotting utilities for feature analysis and model performance

## Project Structure

```
.
├── src/                        # Source code
│   ├── core/                   # Core functionality
│   │   ├── config.py           # Configuration parameters and constants
│   │   └── preprocessing.py    # Data preprocessing and cleaning
│   ├── features/               # Feature engineering and analysis
│   │   ├── analysis.py         # Feature importance and analysis
│   │   ├── selection.py        # Feature selection algorithms
│   │   └── stability.py        # Feature stability measurement (PSI)
│   ├── models/                 # Model training and deployment
│   │   ├── training.py         # Core model training functionality
│   │   ├── hyperopt_tuning.py  # Hyperparameter optimization
│   │   └── deployment.py       # Model deployment utilities
│   ├── evaluation/             # Model evaluation
│   │   └── metrics.py          # Evaluation metrics and scoring
│   ├── visualization/          # Visualization utilities
│   │   ├── plots.py            # Common plotting functions
│   │   └── feature_selection_viz.py # Feature selection visualizations
│   ├── utils/                  # Utility functions
│   │   ├── common.py           # Common utilities
│   │   ├── fonts.py            # Font configuration for plots
│   │   └── imports.py          # Common imports
│   └── main.py                 # Main executable script
├── funnel_models/              # Generated directory for model artifacts
├── tuned_models/               # Generated directory for tuned models
├── optimization_results/       # Generated directory for optimization results
├── deployment_results/         # Generated directory for deployment results
├── requirements.txt            # Project dependencies
├── README.md                   # This README file
└── LICENSE                     # License file
```

## Installation

1. Clone this repository:

   ```
   git clone <repository_url>
   cd customer-conversion-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

The system provides a command-line interface through `src/main.py` with several modes of operation:

### Data Validation

Check for duplicate features across files:

```
python -m src.main --mode check --features <feature_file1> <feature_file2> ...
```

### Merging Feature Files

Merge multiple feature files with sample files:

```
python -m src.main --mode merge --features <feature_file1> <feature_file2> ... --sample <sample_file>
```

### Model Training

Train a model using the two-stage approach:

```
python -m src.main --mode train --data <data_file> --target <target_column>
```

You can also resume training from a specific stage:

```
python -m src.main --mode train --data <data_file> --target <target_column> --resume-from <stage>
```

Where `<stage>` can be one of: `start`, `initial_model`, `feature_analysis`, `feature_selection`, `final_model`.

### Hyperparameter Tuning

Optimize model hyperparameters (requires running training first):

```
python -m src.main --mode tune --target <target_column> --max-evals 50
```

### Model Deployment

Deploy trained model to make predictions:

```
python -m src.main --mode deploy --model <model_file> --data <test_data> --target <target_column> --features-file <feature_list_file>
```

## Model Pipeline

The system uses a two-stage modeling approach:

1. **First Stage**:

   - Train initial model with all features
   - Analyze feature importance using XGBoost native methods
   - Generate SHAP values for deeper feature understanding
   - Evaluate feature stability using PSI across time bins
   - Calculate feature statistics (IV, correlation, missing rate, etc.)
   - Analyze positive sample subgroups for targeted feature selection

2. **Feature Selection**:

   - Multi-criteria feature selection with configurable weights
   - Score features based on importance, stability, information value, and variance
   - Filter highly correlated features using correlation analysis
   - Option to force include or exclude specific features

3. **Second Stage**:

   - Train optimized model with selected features
   - Evaluate model performance with focus on PR-AUC and recall
   - Use adaptive thresholds optimized for F1 or F2 score

4. **Hyperparameter Optimization**:

   - Hyperopt-based tuning with TPE algorithm
   - Optimizes specifically for positive sample prediction
   - Customizable search space defined in config.py
   - Multi-metric evaluation at different threshold levels

5. **Model Deployment**:
   - Batch prediction capabilities
   - Score binning and distribution analysis
   - Consistent feature preprocessing pipeline

## Technology Stack

This project utilizes the following libraries:

- scikit-learn - Machine learning algorithms and utilities
- xgboost - Gradient boosting implementation
- shap - Feature importance explanation
- numpy - Numerical computing
- scipy - Scientific computing
- matplotlib - Data visualization
- seaborn - Statistical data visualization
- pandas - Data manipulation and analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
