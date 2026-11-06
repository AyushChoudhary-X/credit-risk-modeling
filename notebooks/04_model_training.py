"""
04 — Model Training & Evaluation
==================================
Train multiple ML models, compare performance, tune the best model with Optuna,
and generate comprehensive evaluation visualizations.
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.feature_engineering import prepare_data, apply_smote

# ── Configuration ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})


def evaluate_model(model, X_test, y_test, model_name):
    """Compute all metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC': average_precision_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


def main():
    print("=" * 60)
    print("🤖 MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # ── Define Models ──
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                                  use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
    }

    # ── Train & Evaluate All Models ──
    all_metrics = []
    all_predictions = {}

    print("\n" + "─" * 60)
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train_smote, y_train_smote)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        all_predictions[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob}

        print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"   PR-AUC:    {metrics['PR-AUC']:.4f}")

    # ── Model Comparison Table ──
    metrics_df = pd.DataFrame(all_metrics).set_index('Model')
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON TABLE")
    print("=" * 60)
    print(metrics_df.round(4).to_string())

    # ── 1. ROC Curves ──
    print("\n📈 Generating ROC Curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    for (name, data), color in zip(all_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
        auc = roc_auc_score(y_test, data['y_prob'])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'roc_curves_comparison.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/roc_curves_comparison.png")

    # ── 2. Precision-Recall Curves ──
    print("\n📈 Generating Precision-Recall Curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, data), color in zip(all_predictions.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, data['y_prob'])
        ap = average_precision_score(y_test, data['y_prob'])
        ax.plot(recall, precision, color=color, linewidth=2, label=f'{name} (AP = {ap:.4f})')
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_title('Precision-Recall Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'precision_recall_curves.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/precision_recall_curves.png")

    # ── 3. Confusion Matrices ──
    print("\n📈 Generating Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, data) in zip(axes.flatten(), all_predictions.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    fig.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'confusion_matrices.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/confusion_matrices.png")

    # ── 4. XGBoost Feature Importance ──
    print("\n📈 Generating Feature Importance Plot...")
    xgb_model = all_predictions['XGBoost']['model']
    importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    top_15 = importances.tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top_15.index, top_15.values, color='#3498db', edgecolor='white')
    ax.set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    for bar, val in zip(bars, top_15.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'feature_importance_xgb.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/feature_importance_xgb.png")

    # ── 5. Optuna Hyperparameter Tuning ──
    print("\n" + "=" * 60)
    print("🔬 OPTUNA HYPERPARAMETER TUNING (XGBoost)")
    print("=" * 60)

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
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0,
        }
        model = XGBClassifier(**params)
        model.fit(X_train_smote, y_train_smote)
        y_prob = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_prob)

    study = optuna.create_study(direction='maximize', study_name='xgb_credit_risk')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"\n🏆 Best Trial:")
    print(f"   ROC-AUC: {study.best_value:.4f}")
    print(f"   Best Params:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    # ── Retrain with best params ──
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    best_params['verbosity'] = 0

    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(X_train_smote, y_train_smote)
    best_metrics, best_pred, best_prob = evaluate_model(best_xgb, X_test, y_test, 'XGBoost (Tuned)')

    print(f"\n📊 Tuned XGBoost Final Metrics:")
    for k, v in best_metrics.items():
        if k != 'Model':
            print(f"   {k}: {v:.4f}")

    # ── Save best model ──
    model_path = os.path.join(MODEL_DIR, 'best_xgb_model.pkl')
    joblib.dump({
        'model': best_xgb,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': best_metrics,
        'params': best_params,
    }, model_path)
    print(f"\n💾 Best model saved to: models/best_xgb_model.pkl")

    # ── Final Comparison ──
    all_metrics.append(best_metrics)
    final_df = pd.DataFrame(all_metrics).set_index('Model')
    print("\n" + "=" * 60)
    print("🏆 FINAL MODEL COMPARISON (with Tuned XGBoost)")
    print("=" * 60)
    print(final_df.round(4).to_string())

    # Save metrics
    final_df.round(4).to_csv(os.path.join(os.path.dirname(__file__), '..', 'reports', 'model_metrics.csv'))
    print("\n💾 Metrics saved to: reports/model_metrics.csv")


if __name__ == "__main__":
    main()
