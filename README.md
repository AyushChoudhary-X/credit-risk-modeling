# 🏦 Credit Risk Modeling — Loan Default Prediction

## 🎯 Project Overview
A production-grade **credit risk modeling system** that predicts loan defaults using machine learning, with a focus on **regulatory compliance** (RBI-mandated explainability) and **profit optimization** over raw accuracy. The model generates SHAP-based reason codes for every denial decision and uses a custom cost matrix to optimize the probability threshold for maximum bank profitability.

## 🛠️ Tech Stack
- **Python 3.13** — Core language
- **pandas, NumPy** — Data manipulation & analysis
- **scikit-learn** — ML models, preprocessing, metrics
- **XGBoost, LightGBM** — Gradient boosting classifiers
- **SHAP** — Model explainability & reason code generation
- **Optuna** — Bayesian hyperparameter optimization
- **imbalanced-learn** — SMOTE for class imbalance handling
- **matplotlib, seaborn** — Data visualization

## 📁 Project Structure
```text
credit-risk-modeling/
├── data/
│   ├── raw/                    # Original dataset (not tracked)
│   └── processed/              # Cleaned & engineered features
├── notebooks/
│   ├── 01_data_loading_and_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_model_training.py
│   ├── 05_shap_explainability.py
│   └── 06_profit_optimization.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── explainability.py
│   └── profit_optimizer.py
├── models/                     # Serialized model artifacts
├── images/                     # Generated visualizations
├── reports/                    # Analysis reports
├── requirements.txt
└── README.md
```

## 📦 Dataset
- **Source:** [Kaggle — Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Size:** 32,581 records, 12 features
- **License:** CC0 (Public Domain)
- **Target Variable:** `loan_status` (0 = Non-Default, 1 = Default)

### Features
| Feature | Description | Type |
|---------|-------------|------|
| `person_age` | Borrower's age | Numeric |
| `person_income` | Annual income | Numeric |
| `person_home_ownership` | RENT, OWN, MORTGAGE, OTHER | Categorical |
| `person_emp_length` | Employment length (years) | Numeric |
| `loan_intent` | Purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION | Categorical |
| `loan_grade` | Risk grade: A (safest) to G (riskiest) | Ordinal |
| `loan_amnt` | Loan amount requested | Numeric |
| `loan_int_rate` | Interest rate (%) | Numeric |
| `loan_percent_income` | Loan as % of annual income | Numeric |
| `cb_person_default_on_file` | Historical default (Y/N) | Binary |
| `cb_person_cred_hist_length` | Credit history length (years) | Numeric |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation
```bash
git clone https://github.com/AyushChoudhary-X/credit-risk-modeling.git
cd credit-risk-modeling

pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Step 1: Data Loading & Cleaning
python notebooks/01_data_loading_and_cleaning.py

# Step 2: Exploratory Data Analysis
python notebooks/02_eda.py

# Step 3: Feature Engineering
python notebooks/03_feature_engineering.py

# Step 4: Model Training & Evaluation
python notebooks/04_model_training.py

# Step 5: SHAP Explainability & Reason Codes
python notebooks/05_shap_explainability.py

# Step 6: Profit Optimization
python notebooks/06_profit_optimization.py
```

## 📊 Key Results
*Results will be populated after running the pipeline.*

## 📝 License
This project is licensed under the MIT License.
