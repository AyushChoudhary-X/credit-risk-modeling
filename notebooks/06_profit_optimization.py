"""
06 — Profit Optimization & Cost-Sensitive Threshold Selection
==============================================================
Build a custom cost matrix that reflects real banking economics:
- False Positive (approve a defaulter) costs ₹5,00,000 (principal lost)
- False Negative (reject a good customer) costs ₹50,000 (lost interest margin)

Optimize the probability threshold to maximize net profit, not just F1 or accuracy.
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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.feature_engineering import prepare_data

# ── Configuration ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_xgb_model.pkl')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

# ── Cost Matrix (Indian Banking Context) ──
COST_FP = 500000   # ₹5,00,000 — principal lost when approving a defaulter
COST_FN = 50000    # ₹50,000 — lost interest margin from rejecting a good customer
COST_TP = 0        # Correctly identifying a defaulter — no cost
COST_TN = 75000    # ₹75,000 — average interest earned from a good loan (profit)


def calculate_profit(y_true, y_pred):
    """Calculate net profit based on the cost matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    profit_from_good_loans = tn * COST_TN       # Interest earned from correctly approved good loans
    loss_from_defaults = fp * COST_FP            # Principal lost from approved defaulters
    opportunity_cost = fn * COST_FN              # Lost interest from rejected good customers
    savings_from_caught = tp * COST_TP           # No direct profit, but avoided loss

    net_profit = profit_from_good_loans - loss_from_defaults - opportunity_cost + savings_from_caught
    return net_profit, tn, fp, fn, tp


def main():
    print("=" * 60)
    print("💰 PROFIT OPTIMIZATION & COST-SENSITIVE ANALYSIS")
    print("=" * 60)

    # Load model and data
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    feature_names = artifact['feature_names']

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(df)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"✅ Loaded model and {len(X_test):,} test samples")
    print(f"\n💵 Cost Matrix:")
    print(f"   False Positive (approve defaulter):     -₹{COST_FP:,}")
    print(f"   False Negative (reject good customer):  -₹{COST_FN:,}")
    print(f"   True Negative (approve good customer):  +₹{COST_TN:,}")
    print(f"   True Positive (catch defaulter):         ₹{COST_TP:,}")
    print(f"   Cost Ratio: FP is {COST_FP // COST_FN}x more expensive than FN")

    # ── 1. Threshold Sweep ──
    print("\n" + "─" * 60)
    print("📊 THRESHOLD OPTIMIZATION")
    print("─" * 60)

    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        net_profit, tn, fp, fn, tp = calculate_profit(y_test, y_pred)

        results.append({
            'threshold': round(thresh, 2),
            'net_profit': net_profit,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'approved': tn + fp,
            'denied': fn + tp,
            'default_rate_approved': fp / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        })

    results_df = pd.DataFrame(results)

    # Find optimal thresholds
    profit_optimal_idx = results_df['net_profit'].idxmax()
    f1_optimal_idx = results_df['f1'].idxmax()
    default_idx = results_df[results_df['threshold'] == 0.50].index[0]

    profit_optimal = results_df.loc[profit_optimal_idx]
    f1_optimal = results_df.loc[f1_optimal_idx]
    default_result = results_df.loc[default_idx]

    print(f"\n🏆 Optimal Thresholds Found:")
    print(f"\n   📈 Profit-Maximizing Threshold: {profit_optimal['threshold']:.2f}")
    print(f"      Net Profit:  ₹{profit_optimal['net_profit']:,.0f}")
    print(f"      Accuracy:    {profit_optimal['accuracy']:.4f}")
    print(f"      Precision:   {profit_optimal['precision']:.4f}")
    print(f"      Recall:      {profit_optimal['recall']:.4f}")
    print(f"      F1-Score:    {profit_optimal['f1']:.4f}")

    print(f"\n   📊 F1-Maximizing Threshold: {f1_optimal['threshold']:.2f}")
    print(f"      Net Profit:  ₹{f1_optimal['net_profit']:,.0f}")
    print(f"      F1-Score:    {f1_optimal['f1']:.4f}")

    print(f"\n   ⚖️  Default Threshold (0.50):")
    print(f"      Net Profit:  ₹{default_result['net_profit']:,.0f}")
    print(f"      F1-Score:    {default_result['f1']:.4f}")

    profit_gain = profit_optimal['net_profit'] - default_result['net_profit']
    print(f"\n   💡 Profit gain from optimization: ₹{profit_gain:,.0f} ({profit_gain/abs(default_result['net_profit'])*100:+.1f}%)")

    # ── 2. Profit Curve Plot ──
    print("\n📈 Generating Profit Curve...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(results_df['threshold'], results_df['net_profit'] / 1e6, color='#2ecc71', linewidth=2.5, label='Net Profit')
    ax1.axvline(x=profit_optimal['threshold'], color='#2ecc71', linestyle='--', linewidth=1.5,
                label=f'Profit Optimal ({profit_optimal["threshold"]:.2f})')
    ax1.axvline(x=0.50, color='#e74c3c', linestyle=':', linewidth=1.5, label='Default (0.50)')
    ax1.axvline(x=f1_optimal['threshold'], color='#3498db', linestyle='-.', linewidth=1.5,
                label=f'F1 Optimal ({f1_optimal["threshold"]:.2f})')
    ax1.set_xlabel('Probability Threshold', fontsize=12)
    ax1.set_ylabel('Net Profit (₹ Millions)', fontsize=12, color='#2ecc71')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.set_title('Profit Optimization — Finding the Ideal Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['f1'], color='#3498db', linewidth=1.5, alpha=0.7, linestyle='-')
    ax2.set_ylabel('F1-Score', fontsize=12, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'profit_curve.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/profit_curve.png")

    # ── 3. Threshold Comparison Table ──
    print("\n📈 Generating Threshold Comparison Chart...")
    comparison_data = {
        'Metric': ['Threshold', 'Net Profit', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
                    'Approved', 'Denied', 'Default Rate in Approved'],
        'Default (0.50)': [
            '0.50', f'₹{default_result["net_profit"]:,.0f}',
            f'{default_result["accuracy"]:.4f}', f'{default_result["precision"]:.4f}',
            f'{default_result["recall"]:.4f}', f'{default_result["f1"]:.4f}',
            f'{int(default_result["approved"]):,}', f'{int(default_result["denied"]):,}',
            f'{default_result["default_rate_approved"]:.2f}%',
        ],
        'Profit-Optimal': [
            f'{profit_optimal["threshold"]:.2f}', f'₹{profit_optimal["net_profit"]:,.0f}',
            f'{profit_optimal["accuracy"]:.4f}', f'{profit_optimal["precision"]:.4f}',
            f'{profit_optimal["recall"]:.4f}', f'{profit_optimal["f1"]:.4f}',
            f'{int(profit_optimal["approved"]):,}', f'{int(profit_optimal["denied"]):,}',
            f'{profit_optimal["default_rate_approved"]:.2f}%',
        ],
    }
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # ── 4. Risk Bucketing ──
    print("\n" + "─" * 60)
    print("🏷️  RISK BUCKETING — Applicant Categorization")
    print("─" * 60)

    # Define risk bands based on profit-optimal threshold
    opt_thresh = profit_optimal['threshold']
    low_thresh = max(0.10, opt_thresh - 0.20)
    high_thresh = min(0.90, opt_thresh + 0.15)

    def categorize_risk(prob):
        if prob < low_thresh:
            return 'APPROVE'
        elif prob < high_thresh:
            return 'MANUAL REVIEW'
        else:
            return 'REJECT'

    risk_categories = pd.Series(y_prob).apply(categorize_risk)
    risk_counts = risk_categories.value_counts()

    print(f"\n   Risk Bands:")
    print(f"   APPROVE:       P(Default) < {low_thresh:.2f}")
    print(f"   MANUAL REVIEW: {low_thresh:.2f} ≤ P(Default) < {high_thresh:.2f}")
    print(f"   REJECT:        P(Default) ≥ {high_thresh:.2f}")

    print(f"\n   Distribution:")
    for cat in ['APPROVE', 'MANUAL REVIEW', 'REJECT']:
        count = risk_counts.get(cat, 0)
        pct = count / len(y_prob) * 100
        print(f"   {cat:15s}: {count:,} ({pct:.1f}%)")

    # ── 5. Risk Distribution Histogram ──
    print("\n📈 Generating Risk Score Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, color='#2ecc71', label='Non-Default', density=True, edgecolor='white')
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6, color='#e74c3c', label='Default', density=True, edgecolor='white')
    ax.axvline(x=low_thresh, color='orange', linestyle='--', linewidth=2, label=f'Approve/Review ({low_thresh:.2f})')
    ax.axvline(x=high_thresh, color='red', linestyle='--', linewidth=2, label=f'Review/Reject ({high_thresh:.2f})')
    ax.set_title('Risk Score Distribution with Decision Boundaries', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Default Probability')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'risk_score_distribution.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   💾 Saved: images/risk_score_distribution.png")

    # ── Save Profit Report ──
    results_df.to_csv(os.path.join(REPORT_DIR, 'profit_analysis.csv'), index=False)
    print(f"\n💾 Full profit analysis saved to: reports/profit_analysis.csv")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("📋 PROFIT OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  • Profit-optimal threshold: {profit_optimal['threshold']:.2f} (vs default 0.50)")
    print(f"  • Net profit improvement: ₹{profit_gain:,.0f}")
    print(f"  • Cost ratio: FP is {COST_FP // COST_FN}x costlier than FN")
    print(f"  • Risk bucketing: APPROVE / MANUAL REVIEW / REJECT")
    print(f"  • 2 visualizations + 1 analysis report generated")


if __name__ == "__main__":
    main()
