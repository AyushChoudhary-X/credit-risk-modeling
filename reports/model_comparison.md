# 📊 Model Comparison & Evaluation Report

## 1. Executive Summary
This report documents the benchmark results across 4 diverse machine learning classifiers trained to predict probability of default on loan applications, followed by Optuna Bayesian hyperparameter optimization on the top-performing architecture (**XGBoost**).

Models evaluated:
1. **Logistic Regression** (L2-penalized linear baseline)
2. **Random Forest Classifier** (Ensemble bagging of 200 deep decision trees)
3. **XGBoost Classifier** (Gradient boosting on decision trees with exact greedy split finding)
4. **LightGBM Classifier** (Leaf-wise gradient boosting)
5. **XGBoost (Optuna Tuned)** (Tuned across 9 hyperparameters over 30 trials maximizing ROC-AUC)

---

## 2. Benchmark Metrics (Test Set Evaluation)

Evaluated on an out-of-time/out-of-sample stratified holdout of **6,330 loans** (21.6% actual default prevalence).

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.59% | 51.76% | 78.75% | 0.6246 | 0.8706 | 0.7068 |
| **Random Forest** | 91.50% | 83.70% | 75.24% | 0.7924 | 0.9272 | 0.8756 |
| **LightGBM** | 93.59% | 96.15% | 73.19% | 0.8311 | 0.9421 | 0.8927 |
| **XGBoost (Default)** | 93.81% | 97.00% | 73.55% | 0.8367 | 0.9438 | 0.8953 |
| **XGBoost (Tuned)** 🏆 | **93.70%** | **95.48%** | **74.29%** | **0.8356** | **0.9466** | **0.8991** |

---

## 3. Key Observations & Model Selection Rationale

1. **Tree-Based Ensembles Dominate Linear Models**: Logistic Regression achieved an AUC of 0.8706, but suffered from poor precision (51.76%), resulting in high false alarm rates.
2. **Superior Non-Linear Boundary Separation**: Non-linear interactions between `loan_percent_income`, `loan_grade`, and `income_to_loan_ratio` were effectively captured by gradient boosted trees.
3. **Optuna Bayesian Optimization Gains**:
   - The tuned XGBoost model achieved the highest **ROC-AUC (0.9466)** and **PR-AUC (0.8991)**.
   - Optimal hyperparameter configuration:
     - `max_depth`: 5
     - `learning_rate`: 0.2006
     - `n_estimators`: 400
     - `min_child_weight`: 8
     - `subsample`: 0.8695
     - `colsample_bytree`: 0.8515
     - `gamma`: 0.9458
     - `reg_alpha`: 1.5619 (L1 regularization)
     - `reg_lambda`: 8.8513 (L2 regularization)
   - L1/L2 regularization heavily mitigated overfitting while boosting discrimination across borderline applicants.

---

## 4. Visual Diagnostics
Visual assets generated during evaluation:
- `images/roc_curves_comparison.png`: ROC Curves for all 4 models.
- `images/precision_recall_curves.png`: Precision-Recall Curves highlighting superior AUC-PR under class imbalance.
- `images/confusion_matrices.png`: 2x2 multi-panel confusion matrix breakdown.
- `images/feature_importance_xgb.png`: Top 15 split-importance features.
