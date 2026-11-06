"""
Model Training & Evaluation Module
==================================
Reusable training, hyperparameter optimization, and evaluation routines.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_baseline_models():
    """Return dictionary of initialized baseline models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                                  eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
    }


def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate classifier performance across standard credit risk classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'Model': model_name,
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred)),
        'F1-Score': float(f1_score(y_test, y_pred)),
        'ROC-AUC': float(roc_auc_score(y_test, y_prob)),
        'PR-AUC': float(average_precision_score(y_test, y_prob)),
    }


def tune_xgboost(X_train, y_train, X_test, y_test, n_trials: int = 30):
    """Run Optuna Bayesian hyperparameter search for XGBoost targeting ROC-AUC."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0,
        }
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
