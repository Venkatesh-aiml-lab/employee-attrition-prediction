# Employee Attrition Prediction

## 📌 Project Overview

Employee attrition is a significant HR challenge because employee turnover can increase recruitment costs, affect productivity, and impact workforce planning.

The objective of this project is to analyze employee-related factors associated with attrition and build machine learning classification models that can help identify employees who may be at higher risk of leaving the organization.

This project covers:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering and feature selection
- Machine learning model development
- Stratified cross-validation
- Independent test-set evaluation
- Model comparison
- MLflow experiment tracking
- MLflow model logging and registration
- Champion model aliasing

> **Project Version:** Final working version of the current implementation.

---

## 🎯 Business Objective

The HR team wants to answer two key questions:

1. **What employee-related factors are associated with attrition?**
2. **Can machine learning predict whether an employee is likely to leave the organization?**

The project uses historical employee data to identify patterns in employee attrition and evaluate multiple classification algorithms.

The model is intended as a **decision-support tool** for HR analysis and retention planning, not as an automatic basis for employment decisions.

---

## 📊 Dataset

The dataset contains:

- **1,470 employee records**
- **35 features**
- Target variable: `Attrition`

Target values:

| Value | Meaning |
|---|---|
| `No` | Employee stayed with the organization |
| `Yes` | Employee left the organization |

The target variable is imbalanced:

| Class | Approx. Distribution |
|---|---:|
| No | 83.87% |
| Yes | 16.13% |

Because attrition is an imbalanced classification problem, the project evaluates multiple metrics instead of relying only on accuracy.

---

## 🔎 Exploratory Data Analysis

The EDA includes:

- Dataset shape and structure
- Data types
- Missing-value analysis
- Duplicate-record analysis
- Unique-value analysis
- Numerical and categorical feature analysis
- Target-class distribution
- Distribution analysis
- Correlation analysis
- Correlation heatmaps
- Outlier analysis using the IQR method
- Skewness analysis
- Visualization of selected feature distributions

### Key EDA observations

- The dataset contains 1,470 rows and 35 columns.
- No missing values were identified.
- No duplicate records were identified.
- The target variable is imbalanced.
- `StandardHours`, `Over18`, and `EmployeeCount` contain constant values and do not contribute useful predictive information.
- Several numerical features show skewness and potential outliers.

---

## 🛠️ Data Preprocessing

The current implementation includes the following preprocessing steps:

### 1. Removing non-informative features

The following features are removed because they contain constant or non-predictive information:

```text
EmployeeNumber
Over18
StandardHours
EmployeeCount
```

### 2. Target encoding

The target variable is converted from:

```text
No → 0
Yes → 1
```

### 3. Binary categorical encoding

`Gender` and `OverTime` are label encoded.

### 4. One-hot encoding

The following categorical features are one-hot encoded:

```text
BusinessTravel
Department
EducationField
JobRole
MaritalStatus
```

The encoder uses:

- `drop="first"`
- `handle_unknown="ignore"`

### 5. Yeo-Johnson transformation

Yeo-Johnson transformation is applied to selected skewed numerical features, including:

```text
DistanceFromHome
MonthlyIncome
JobLevel
PerformanceRating
NumCompaniesWorked
TotalWorkingYears
YearsAtCompany
YearsSinceLastPromotion
YearsWithCurrManager
```

### 6. Mutual Information feature selection

`mutual_info_classif` is used to calculate the relationship between input features and the target.

Features with mutual information greater than zero are selected for modeling.

---

## 🔬 Train / Validation / Test Strategy

The project uses a stratified train/test split:

```text
80% → Training
20% → Independent Test
```

```python
train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=50,
    stratify=y
)
```

The training data is further evaluated using **5-fold Stratified Cross-Validation**.

### Validation strategy

```text
Full Dataset
     │
     ├── 80% Training Data
     │      │
     │      └── 5-Fold Stratified Cross-Validation
     │
     └── 20% Independent Test Data
```

The independent test set is kept separate from the cross-validation process and is used for final model evaluation.

---

## 🤖 Machine Learning Models

Three classification algorithms are evaluated.

### 1. Logistic Regression

Used as a baseline classification model.

Configuration:

```text
max_iter = 10000
```

### 2. Random Forest

An ensemble tree-based classification model.

Configuration:

```text
n_estimators = 100
max_depth = 6
random_state = 50
```

### 3. XGBoost

A gradient-boosting classification model.

Configuration:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 6
random_state = 50
eval_metric = "auc"
```

---

## 📈 Model Evaluation

The models are evaluated using:

- Validation Accuracy
- Validation ROC-AUC
- Test Accuracy
- Test Precision
- Test Recall
- Test F1-score
- Test ROC-AUC
- Confusion Matrix
- ROC Curves

### Current Results

| Model | CV Accuracy | CV AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 86.57% | 0.807 | 86.73% | 0.70 | 0.418 | 0.418 | **0.807** |
| Random Forest | 85.29% | 0.768 | 87.07% | 1.00 | 0.191 | 0.321 | 0.795 |
| XGBoost | 85.97% | 0.760 | **87.42%** | 0.75 | 0.319 | **0.448** | 0.787 |

### Current model selection

The project selects the model with the highest **test accuracy** for the MLflow Model Registry.

Based on the current results:

**XGBoost is the selected model with a test accuracy of 87.42%.**

The model comparison and evaluation artifacts are logged to MLflow for reproducibility.

> Note: Because the target is imbalanced, model selection based only on accuracy has limitations. Recall, F1-score and ROC-AUC should also be considered when the business priority is identifying employees at risk of attrition.

---

## 📊 Model Visualizations

The project generates and logs visualizations including:

- Confusion matrices
- ROC curves
- Feature-importance plots
- Model accuracy comparison
- Model AUC comparison
- Combined ROC comparison
- Mutual Information feature scores

These artifacts are available through the MLflow experiment tracking workflow.

---

# 🔬 MLflow Integration

MLflow is used to track experiments, metrics, parameters, artifacts, and trained models.

### MLflow Experiment

```text
employee_attrition
```

### Models logged

```text
Logistic Regression
Random Forest
XGBoost
```

### Metrics logged

For each model, MLflow tracks:

- Validation accuracy mean
- Validation accuracy standard deviation
- Validation AUC mean
- Validation AUC standard deviation
- Test accuracy
- Test precision
- Test recall
- Test F1-score
- Test AUC

### Artifacts logged

Examples include:

```text
ROC curves
Confusion matrices
Feature-importance plots
Model comparison CSV
```

---

## 🗂️ MLflow Model Registry

The best-performing model based on the configured selection metric is registered using the model name:

```text
Best_Attrition_Prediction_Model
```

The selected model is assigned the alias:

```text
champion
```

The registry workflow includes:

1. Searching MLflow experiment runs
2. Comparing model test accuracy
3. Identifying the best model run
4. Finding the corresponding logged model
5. Registering the model
6. Adding model-version tags
7. Assigning the `champion` alias
8. Loading the champion model for verification
9. Generating sample predictions

---

## ☁️ Google Colab and MLflow Dashboard

The project is designed to run in **Google Colab**.

The notebook:

- Installs MLflow, pyngrok, and XGBoost
- Starts an MLflow tracking server
- Uses SQLite as the MLflow backend store
- Stores MLflow artifacts
- Creates an ngrok tunnel
- Provides a public MLflow dashboard URL for the current Colab runtime

The MLflow dashboard URL is generated dynamically when the notebook runs.

> The ngrok authentication token is requested securely at runtime and is not stored in the notebook source code.

### Important

The MLflow dashboard URL is temporary and depends on the active Colab runtime. The Colab runtime must remain active while using the dashboard.

---

## 📁 Project Structure

```text
employee-attrition-prediction/
│
├── code/
│   ├── employee_attrition.ipynb
│   └── employee_attrition.py
│
├── data/
│   └── data_employee_attrition.csv
│
├── documents/
│   ├── requirements.docx
│   ├── Solution flow for Employee Attrition.drawio
│   └── Employee_Attrition_MLflow_Issues_and_Resolutions.docx
│
├── artifacts/
│   ├── mlflow_registry_v1.png
│   ├── mlflow_runs.png
│   ├── mlflow_model_xgboost.png
│   ├── mlflow_model_registry.png
│   └── mlflow_models.png
│
├── mlflow_plots/
│   ├── logistic_regression_confusion_matrix.png
│   ├── logistic_regression_roc.png
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_feature_importance.png
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_feature_importance.png
│   └── model_comparison.csv
│
├── .gitignore
└── README.md
```

### Directory Description

| Directory/File | Purpose |
|---|---|
| `code/` | Main notebook and Python implementation |
| `data/` | Employee attrition dataset |
| `documents/` | Requirements, solution flow and MLflow troubleshooting documentation |
| `artifacts/` | MLflow-related screenshots and registry evidence |
| `mlflow_plots/` | Model evaluation visualizations and comparison results |
| `.gitignore` | Prevents temporary files, environments and MLflow local artifacts from being committed |
| `README.md` | Project documentation |

> The local `mlflow.db` database and `mlartifacts/` directory are runtime-generated MLflow files and are excluded by `.gitignore`.

---

## 🧰 Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Experiment Tracking / MLOps

- MLflow
- MLflow Model Registry
- pyngrok

### Development Environment

- Google Colab
- Jupyter Notebook
- Git
- GitHub

### Documentation / Design

- Draw.io

---

## ▶️ How to Run

### Option 1 — Google Colab

1. Clone or download the repository.
2. Open:

```text
code/employee_attrition.ipynb
```

3. Open the notebook in Google Colab.
4. Install the required packages by running the setup cells.
5. Provide the required ngrok authentication token when prompted.
6. Make sure the dataset is available at the path expected by the notebook.
7. Run the notebook cells sequentially.
8. Open the MLflow dashboard URL printed by the notebook.

### Option 2 — Python Script

The project also contains:

```text
code/employee_attrition.py
```

The Python script contains the core implementation of the project.

The MLflow/Google Colab configuration in the current implementation is designed around the Colab environment, so local execution may require path and environment adjustments.

---

## 📦 Requirements

The project requirements are documented in:

```text
documents/requirements.docx
```

The MLflow experiment environment also records model-specific dependency information as part of the logged model artifacts.

---

## 🔄 End-to-End Workflow

```text
Employee Dataset
       │
       ▼
Data Understanding
       │
       ▼
Data Quality Checks
       │
       ├── Missing Values
       ├── Duplicate Records
       ├── Unique Values
       └── Target Distribution
       │
       ▼
Exploratory Data Analysis
       │
       ├── Univariate Analysis
       ├── Bivariate Analysis
       ├── Correlation Analysis
       ├── Outlier Analysis
       └── Skewness Analysis
       │
       ▼
Feature Engineering
       │
       ├── Remove Non-informative Features
       ├── Encode Categorical Features
       ├── Yeo-Johnson Transformation
       └── Mutual Information Selection
       │
       ▼
Train / Test Split
       │
       ├── 80% Training
       │      └── 5-Fold Stratified CV
       │
       └── 20% Independent Test
       │
       ▼
Model Training
       │
       ├── Logistic Regression
       ├── Random Forest
       └── XGBoost
       │
       ▼
Model Evaluation
       │
       ├── Accuracy
       ├── Precision
       ├── Recall
       ├── F1
       ├── ROC-AUC
       ├── Confusion Matrix
       └── ROC Curves
       │
       ▼
Model Comparison
       │
       ▼
Best Model Selection
       │
       ▼
MLflow Tracking
       │
       ├── Parameters
       ├── Metrics
       ├── Artifacts
       └── Logged Models
       │
       ▼
MLflow Model Registry
       │
       └── Best_Attrition_Prediction_Model
                    │
                    ▼
               champion Alias
```

---

## 💡 Business Use Cases

The project can support HR teams in:

- Identifying patterns associated with employee attrition
- Understanding factors related to employee turnover
- Identifying employees who may require further retention analysis
- Supporting workforce planning
- Prioritizing employee-retention initiatives
- Providing data-driven insights for HR analysis

The predictions should be combined with appropriate HR processes and human judgment.

---

## 🚀 Future Improvements

Potential future improvements include:

- Addressing class imbalance using appropriate techniques
- Building a complete preprocessing pipeline
- Moving preprocessing and feature selection inside the cross-validation pipeline
- Hyperparameter tuning
- Threshold optimization for attrition-risk prediction
- Improved model selection based on business-oriented metrics
- SHAP-based model explainability
- Model deployment through an API
- CI/CD integration
- Data and model versioning
- Model monitoring
- Data drift and model drift detection
- Automated retraining
- Production model serving

---

## 📌 Project Status

**Status:** Final working version of the current implementation

**Domain:** Human Resources / People Analytics

**Problem Type:** Binary Classification

**Target:** Employee Attrition

**Best Model by Current Selection Metric:** XGBoost

**Current Test Accuracy:** 87.42%

**MLflow Model Registry:** Implemented

**Champion Model Alias:** Implemented

---

## 👤 Author

**Venkatesh**

This project represents practical work and continued exploration in **Machine Learning, Data Science, and MLOps**, with a focus on applying machine learning and experiment tracking to an HR analytics use case.
