# Customer Conversion Prediction Model

This repository contains a machine learning system specifically designed for predicting customer conversion events, with a focus on optimizing for positive sample prediction. The system includes a full pipeline from data preprocessing to hyperparameter tuning and model deployment.

## Features

- **Two-stage modeling approach**: Initial model for feature selection followed by optimized model for prediction
- **Focus on positive sample prediction**: Specifically optimized for PR-AUC metrics and positive sample recall
- **Feature importance analysis**: Multiple methods for understanding feature contributions (XGBoost importance, SHAP values)
- **Feature stability analysis**: Population Stability Index (PSI) to detect unstable features
- **Comprehensive hyperparameter optimization**: Using Hyperopt for finding optimal model parameters
- **Model deployment utilities**: Easy deployment with consistent feature preprocessing
- **Modular architecture**: Well-organized code for maintainability and extensibility

## Project Structure

```
.
├── src/                        # Source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration parameters and constants
│   ├── preprocessing.py        # Data preprocessing and cleaning
│   ├── feature_engineering.py  # Feature selection and analysis
│   ├── modeling.py             # Core model training functionality
│   ├── hyperopt_tuning.py      # Hyperparameter optimization
│   ├── deployment.py           # Model deployment utilities
│   ├── utils.py                # Utility functions
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

Check for duplicate features across files or duplicate keys within files:

```
python -m src.main --mode check --features <feature_file1> <feature_file2> ...
```

### Merging Feature Files

Merge multiple feature files and process with sample files:

```
python -m src.main --mode merge --features <feature_file1> <feature_file2> ... --sample1 <sample_file1> --sample2 <sample_file2>
```

### Model Training

Train a model using the two-stage approach:

```
python -m src.main --mode train --data <data_file> --target <target_column>
```

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
   - Analyze feature importance
   - Evaluate feature stability using PSI
   - Calculate feature statistics (IV, correlation, missing rate, etc.)

2. **Feature Selection**:

   - Identify important features for positive sample prediction using SHAP values
   - Trim features using multiple criteria (importance, stability, statistics)
   - Filter highly correlated features

3. **Second Stage**:

   - Train optimized model with selected features
   - Evaluate model performance with focus on PR-AUC and recall

4. **Model Deployment**:
   - Export model in PMML format for easy deployment
   - Provide utilities for scoring new data
   - Ensure consistent feature preprocessing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
