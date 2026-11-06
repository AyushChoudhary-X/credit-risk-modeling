"""
02 — Exploratory Data Analysis
================================
Comprehensive EDA of the cleaned credit risk dataset.
Generates 10 publication-quality visualizations.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11})
COLORS = ['#2ecc71', '#e74c3c']


def save_fig(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   💾 Saved: images/{name}")


def main():
    print("=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} records\n")

    # ── 1. Target Distribution ──
    print("1️⃣  Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['loan_status'].value_counts()
    bars = ax.bar(['Non-Default (0)', 'Default (1)'], [counts[0], counts[1]], color=COLORS, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, [counts[0], counts[1]]):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}\n({pct:.1f}%)', ha='center', fontweight='bold')
    ax.set_title('Loan Default Distribution (Class Imbalance)')
    ax.set_ylabel('Count')
    save_fig(fig, 'target_distribution.png')
    print(f"   Non-Default: {counts[0]:,} ({counts[0]/len(df)*100:.1f}%) | Default: {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")

    # ── 2. Numeric Feature Distributions ──
    print("\n2️⃣  Numeric Feature Distributions")
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.hist(df[col], bins=40, color='#3498db', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontweight='bold')
        ax.set_ylabel('Count')
        median_val = df[col].median()
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
        ax.legend(fontsize=8)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Distribution of Numeric Features', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'numeric_distributions.png')

    # ── 3. Default Rate by Loan Grade ──
    print("\n3️⃣  Default Rate by Loan Grade")
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_default = df.groupby('loan_grade')['loan_status'].mean().reindex(grade_order) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_grade = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(grade_order)))
    bars = ax.bar(grade_order, grade_default.values, color=colors_grade, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, grade_default.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_title('Default Rate by Loan Grade')
    ax.set_xlabel('Loan Grade (A = Safest → G = Riskiest)')
    ax.set_ylabel('Default Rate (%)')
    save_fig(fig, 'default_rate_by_grade.png')
    for g, r in grade_default.items():
        print(f"   Grade {g}: {r:.1f}%")

    # ── 4. Default Rate by Loan Intent ──
    print("\n4️⃣  Default Rate by Loan Intent")
    intent_default = df.groupby('loan_intent')['loan_status'].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_intent = plt.cm.viridis(np.linspace(0.2, 0.8, len(intent_default)))
    bars = ax.barh(intent_default.index, intent_default.values, color=colors_intent, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, intent_default.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold')
    ax.set_title('Default Rate by Loan Purpose')
    ax.set_xlabel('Default Rate (%)')
    fig.tight_layout()
    save_fig(fig, 'default_rate_by_intent.png')
    for intent, rate in intent_default.items():
        print(f"   {intent}: {rate:.1f}%")

    # ── 5. Default Rate by Home Ownership ──
    print("\n5️⃣  Default Rate by Home Ownership")
    own_default = df.groupby('person_home_ownership')['loan_status'].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(own_default.index, own_default.values, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, own_default.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_title('Default Rate by Home Ownership Status')
    ax.set_ylabel('Default Rate (%)')
    save_fig(fig, 'default_rate_by_ownership.png')

    # ── 6. Correlation Heatmap ──
    print("\n6️⃣  Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols + ['loan_status']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap')
    fig.tight_layout()
    save_fig(fig, 'correlation_heatmap.png')
    # Top correlations with target
    target_corr = corr['loan_status'].drop('loan_status').abs().sort_values(ascending=False)
    print("   Top correlations with loan_status:")
    for feat, val in target_corr.head(5).items():
        print(f"   {feat}: {corr['loan_status'][feat]:.3f}")

    # ── 7. Income by Default Status ──
    print("\n7️⃣  Income Distribution by Default Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [df[df['loan_status']==0]['person_income'], df[df['loan_status']==1]['person_income']]
    bp = ax.boxplot(data_to_plot, labels=['Non-Default', 'Default'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS[0])
    bp['boxes'][1].set_facecolor(COLORS[1])
    ax.set_title('Income Distribution by Loan Status')
    ax.set_ylabel('Annual Income')
    save_fig(fig, 'income_vs_loan_boxplot.png')
    print(f"   Non-Default median income: ${df[df['loan_status']==0]['person_income'].median():,.0f}")
    print(f"   Default median income: ${df[df['loan_status']==1]['person_income'].median():,.0f}")

    # ── 8. Loan % Income by Status ──
    print("\n8️⃣  Loan Percent Income Distribution by Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color, label in [(0, COLORS[0], 'Non-Default'), (1, COLORS[1], 'Default')]:
        subset = df[df['loan_status'] == status]['loan_percent_income']
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True, edgecolor='white')
    ax.set_title('Loan-to-Income Ratio Distribution by Default Status')
    ax.set_xlabel('Loan as % of Income')
    ax.set_ylabel('Density')
    ax.legend()
    save_fig(fig, 'loan_percent_income_by_status.png')

    # ── 9. Default Rate by Credit History ──
    print("\n9️⃣  Default Rate by Prior Default History")
    hist_default = df.groupby('cb_person_default_on_file')['loan_status'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['No Prior Default (N)', 'Prior Default (Y)'], hist_default.reindex(['N', 'Y']).values,
                  color=[COLORS[0], COLORS[1]], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, hist_default.reindex(['N', 'Y']).values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax.set_title('Default Rate by Prior Default History')
    ax.set_ylabel('Default Rate (%)')
    save_fig(fig, 'default_rate_by_history.png')
    print(f"   No prior default: {hist_default['N']:.1f}% | Prior default: {hist_default['Y']:.1f}%")

    # ── 10. Age Distribution by Status ──
    print("\n🔟 Age Distribution by Default Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color, label in [(0, COLORS[0], 'Non-Default'), (1, COLORS[1], 'Default')]:
        subset = df[df['loan_status'] == status]['person_age']
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True, edgecolor='white')
    ax.set_title('Age Distribution by Default Status')
    ax.set_xlabel('Age')
    ax.set_ylabel('Density')
    ax.legend()
    save_fig(fig, 'age_distribution_by_status.png')

    # ── Summary ──
    print("\n" + "=" * 60)
    print("📋 KEY EDA FINDINGS")
    print("=" * 60)
    print(f"  • Class imbalance: {counts[1]/len(df)*100:.1f}% default rate (needs SMOTE)")
    print(f"  • Loan grade is highly predictive — Grade G default rate: {grade_default.get('G', 0):.1f}%")
    print(f"  • Prior default history strongly predicts future default")
    print(f"  • Higher loan_percent_income correlates with higher default risk")
    print(f"  • Defaulters tend to have lower income and shorter credit histories")
    print(f"  • All 10 visualizations saved to images/")


if __name__ == "__main__":
    main()
