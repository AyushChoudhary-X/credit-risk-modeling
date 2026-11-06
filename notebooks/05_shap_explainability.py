"""
05 — SHAP Explainability & RBI-Compliant Reason Codes
======================================================
Generate model explanations using SHAP (SHapley Additive exPlanations).
Produces global feature importance, dependence plots, individual explanations,
and RBI-compliant denial reason codes for each rejected applicant.
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import prepare_data

# ── Configuration ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_xgb_model.pkl')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_reason_codes(shap_values_array, feature_names, X_test, y_pred, y_prob, top_n=3):
    """
    Generate RBI-compliant denial reason codes.
    For each denied applicant (predicted default), extract the top N features
    contributing most to the denial decision.
    """
    denied_mask = y_pred == 1
    denied_indices = np.where(denied_mask)[0]

    reason_records = []
    for idx in denied_indices:
        sv = shap_values_array[idx]
        # Sort by absolute SHAP value (most impactful features first)
        sorted_indices = np.argsort(-np.abs(sv))

        reasons = []
        for rank, feat_idx in enumerate(sorted_indices[:top_n], 1):
            feat_name = feature_names[feat_idx]
            shap_val = sv[feat_idx]
            feat_val = X_test.iloc[idx, feat_idx]
            direction = "increased" if shap_val > 0 else "decreased"
            reasons.append(f"{feat_name} ({direction} risk)")

        reason_records.append({
            'applicant_index': X_test.index[idx],
            'default_probability': round(y_prob[idx], 4),
            'risk_score': round(y_prob[idx] * 100, 1),
            'reason_1': reasons[0] if len(reasons) > 0 else '',
            'reason_2': reasons[1] if len(reasons) > 1 else '',
            'reason_3': reasons[2] if len(reasons) > 2 else '',
            'reason_code_summary': '; '.join(reasons),
        })

    return pd.DataFrame(reason_records)


def main():
    print("=" * 60)
    print("🔍 SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Load model and data
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    feature_names = artifact['feature_names']
    print(f"✅ Loaded tuned XGBoost model")
    print(f"   ROC-AUC: {artifact['metrics']['ROC-AUC']:.4f}")

    # Prepare test data
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(df)

    # ── Compute SHAP Values ──
    print("\n⏳ Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_values_array = shap_values.values
    print(f"✅ SHAP values computed for {len(X_test):,} test samples")

    # ── 1. SHAP Summary Plot (Bar) ──
    print("\n1️⃣  Global Feature Importance (Bar Plot)")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title('SHAP Feature Importance — Global', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'shap_summary_bar.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/shap_summary_bar.png")

    # ── 2. SHAP Beeswarm Plot ──
    print("\n2️⃣  SHAP Beeswarm Plot (Feature Impact Distribution)")
    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title('SHAP Beeswarm Plot — Feature Impact on Default Prediction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'shap_beeswarm.png'), bbox_inches='tight', facecolor='white')
    plt.close('all')
    print("   💾 Saved: images/shap_beeswarm.png")

    # ── 3. SHAP Dependence Plots (Top 4 Features) ──
    print("\n3️⃣  SHAP Dependence Plots (Top 4 Features)")
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    top_features_idx = np.argsort(-mean_abs_shap)[:4]
    top_features = [feature_names[i] for i in top_features_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat_name, feat_idx in zip(axes.flatten(), top_features, top_features_idx):
        ax.scatter(X_test.iloc[:, feat_idx], shap_values_array[:, feat_idx],
                   c=shap_values_array[:, feat_idx], cmap='RdBu_r', alpha=0.5, s=5)
        ax.set_xlabel(feat_name)
        ax.set_ylabel('SHAP Value')
        ax.set_title(f'Dependence: {feat_name}', fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    fig.suptitle('SHAP Dependence Plots — Top 4 Features', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'shap_dependence_plots.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/shap_dependence_plots.png")
    for feat in top_features:
        print(f"   Top feature: {feat}")

    # ── 4. Individual Waterfall Plots (3 Sample Applicants) ──
    print("\n4️⃣  Individual Prediction Explanations (Waterfall Plots)")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Pick: 1 clear approve, 1 clear deny, 1 borderline
    approve_idx = np.argmin(y_prob)
    deny_idx = np.argmax(y_prob)
    borderline_candidates = np.where((y_prob > 0.35) & (y_prob < 0.65))[0]
    borderline_idx = borderline_candidates[0] if len(borderline_candidates) > 0 else len(y_prob) // 2

    sample_cases = [
        (approve_idx, 'Low Risk (Clear Approve)'),
        (deny_idx, 'High Risk (Clear Deny)'),
        (borderline_idx, 'Borderline (Manual Review)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (idx, title) in zip(axes, sample_cases):
        plt.sca(ax)
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        ax.set_title(f'{title}\nP(Default) = {y_prob[idx]:.3f}', fontsize=11, fontweight='bold')
    fig.suptitle('Individual Prediction Explanations — SHAP Waterfall', fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'shap_waterfall_samples.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/shap_waterfall_samples.png")

    for idx, title in sample_cases:
        print(f"   {title}: P(Default) = {y_prob[idx]:.4f}")

    # ── 5. RBI-Compliant Denial Reason Codes ──
    print("\n5️⃣  Generating RBI-Compliant Denial Reason Codes")
    reason_df = generate_reason_codes(shap_values_array, feature_names, X_test, y_pred, y_prob, top_n=3)
    reason_path = os.path.join(REPORT_DIR, 'denial_reason_codes.csv')
    reason_df.to_csv(reason_path, index=False)

    print(f"   Generated reason codes for {len(reason_df):,} denied applicants")
    print(f"   💾 Saved: reports/denial_reason_codes.csv")

    print(f"\n   📋 Sample Denial Reason Codes (first 5):")
    for _, row in reason_df.head(5).iterrows():
        print(f"   ─────────────────────────────────────")
        print(f"   Applicant #{row['applicant_index']} | Risk Score: {row['risk_score']}")
        print(f"   Reason 1: {row['reason_1']}")
        print(f"   Reason 2: {row['reason_2']}")
        print(f"   Reason 3: {row['reason_3']}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("📋 SHAP ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  • Top predictive feature: {top_features[0]}")
    print(f"  • Total denied applicants: {len(reason_df):,}")
    print(f"  • Average risk score for denied: {reason_df['risk_score'].mean():.1f}/100")
    print(f"  • 5 visualizations + 1 reason code report generated")


if __name__ == "__main__":
    main()
