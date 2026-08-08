# Employee Attrition Prediction

## 📌 Project Overview

Employee attrition is an important HR challenge because employee turnover can increase recruitment costs, affect productivity, and impact team performance.

The objective of this project is to analyze employee-related factors and build machine learning models that can help the HR team understand **why employees may leave the organization** and identify employees who may be at higher risk of attrition.

This project covers data exploration, preprocessing, feature analysis, model building, and model evaluation.

> **Note:** This is the initial version of the project. Further improvements to preprocessing, class-imbalance handling, model optimization, and deployment/monitoring are planned for upcoming versions.

---

## 🎯 Business Objective

The HR team wants to answer two key questions:

1. **Why are employees leaving the organization?**
2. **Can machine learning help identify employees who are more likely to leave?**

The project uses historical employee data to identify patterns associated with attrition and evaluates classification models for predicting employee attrition.

---

## 📂 Project Structure

```text
employee_attrision_predictions/
│
├── code/
│   ├── employee_attrition_v1.ipynb
│   └── employee_attrition_v1.py
│
├── data/
│   └── data_employee_attrition.csv
│
├── documents/
│   ├── requirements.docx
│   └── Solution flow for Employee Attrition.drawio
│
└── README.md
```

### Folder Description

| Folder/File | Description |
|---|---|
| `code/` | Jupyter Notebook and Python source code |
| `data/` | Employee attrition dataset used for analysis and modeling |
| `documents/` | Project requirements and solution flow diagram |
| `employee_attrition_v1.ipynb` | Complete exploratory analysis and ML workflow |
| `employee_attrition_v1.py` | Python version of the project code |
| `README.md` | Project documentation |

---

## 📊 Dataset

The dataset contains **1,470 employee records and 35 features**.

The target variable is:

- `Attrition`
  - `Yes` — employee left the organization
  - `No` — employee stayed with the organization

The target distribution is approximately:

- **No:** 83.88%
- **Yes:** 16.12%

This indicates that the target variable is **imbalanced**, which is an important consideration when evaluating classification models.

---

## 🔎 Exploratory Data Analysis

The project performs exploratory data analysis to understand:

- Dataset structure and data types
- Missing values
- Duplicate records
- Numerical and categorical features
- Target-variable distribution
- Statistical characteristics of numerical variables
- Relationships between features
- Feature importance/relevance
- Potential outliers
- Factors associated with employee attrition

The analysis is intended to provide HR-oriented insights into the characteristics associated with employee turnover.

---

## 🛠️ Data Preprocessing

The current version includes preprocessing steps such as:

- Removing unnecessary columns
- Handling categorical variables
- Encoding categorical features
- Numerical feature transformation
- Feature analysis
- Preparing data for machine learning
- Train/test data preparation

The preprocessing pipeline will be further refined in future versions.

---

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

### 1. Logistic Regression

Used as a baseline classification model and to understand the relationship between employee features and attrition.

### 2. Random Forest

An ensemble tree-based model used to capture nonlinear relationships between employee characteristics and attrition.

### 3. XGBoost

A gradient-boosting algorithm evaluated for its ability to model complex relationships in the employee data.

---

## 📈 Model Evaluation

The models were evaluated using classification metrics including:

- Accuracy
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve

Current results:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 88.10% | 0.806 |
| Random Forest | 86.39% | **0.826** |
| XGBoost | 86.73% | 0.799 |

Based on ROC-AUC in the current version, **Random Forest performed best among the evaluated models**.

Because employee attrition is an imbalanced classification problem, future versions will place greater emphasis on recall, precision, F1-score, ROC-AUC, and other appropriate evaluation approaches rather than relying primarily on accuracy.

---

## 💡 Business Interpretation

The purpose of this project is not simply to predict attrition.

The broader objective is to help HR understand patterns that may be associated with employee turnover.

Potential applications include:

- Identifying employees at higher attrition risk
- Understanding important employee-related factors
- Supporting employee-retention strategies
- Helping HR prioritize further investigation
- Supporting data-driven workforce planning

The model output should be treated as a **decision-support tool**, not as an automatic basis for employment decisions.

---

## 🔄 Current Project Workflow

```text
Employee Data
      ↓
Data Understanding
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Analysis
      ↓
Train / Test Split
      ↓
Model Training
      ↓
Logistic Regression
Random Forest
XGBoost
      ↓
Model Evaluation
      ↓
Compare Model Performance
      ↓
Identify Better Performing Model
```

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook / Google Colab
- Draw.io

---

## ▶️ How to Run the Project

### Option 1: Jupyter Notebook / Google Colab

1. Clone or download this repository.
2. Open:

```text
code/employee_attrition_v1.ipynb
```

3. Upload or place the dataset in the expected data location.
4. Install the required Python libraries.
5. Run the notebook cells sequentially.

### Option 2: Python Script

Run:

```bash
python code/employee_attrition_v1.py
```

Make sure the required Python dependencies are installed before execution.

---

## 📦 Requirements

The required packages are documented in:

```text
documents/requirements.docx
```

A dedicated `requirements.txt` file will be added in a future version.

---

## 🚀 Future Improvements

The project is currently **Version 1**. Planned improvements include:

- Better handling of class imbalance
- Improved preprocessing pipeline
- Prevention of data leakage
- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Improved feature selection
- Threshold optimization
- Model explainability
- SHAP-based feature interpretation
- MLflow experiment tracking
- Model versioning
- Model deployment
- API integration
- Model monitoring
- Data/model drift monitoring
- Automated retraining pipeline

---

## 📌 Project Status

**Version:** 1.0  
**Status:** Initial machine learning implementation

This repository represents the initial version of the Employee Attrition Prediction project. The project will evolve through subsequent versions as additional ML engineering and MLOps capabilities are implemented.

---

## 👤 Author

**Venkatesh**

This project is part of my ongoing exploration and practical work in **Machine Learning, Data Science, and MLOps**.
