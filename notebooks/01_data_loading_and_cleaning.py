"""
01 — Data Loading & Cleaning
=============================
Load the raw credit risk dataset, perform data quality checks,
handle missing values and outliers, and save the cleaned dataset.

Dataset: Kaggle Credit Risk Dataset (laotse/credit-risk-dataset)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import clean_data

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'credit_risk_dataset.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')


def main():
    """Run the data cleaning pipeline."""
    df = clean_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("📊 CLEANED DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Target Variable Distribution:")
    target_counts = df['loan_status'].value_counts()
    total = len(df)
    for label, count in target_counts.items():
        status = "Non-Default" if label == 0 else "Default"
        print(f"   {status} ({label}): {count:,} ({count/total*100:.1f}%)")
    
    print(f"\n📈 Numeric Feature Statistics:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe().round(2).to_string())
    
    print(f"\n📝 Categorical Feature Cardinality:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_vals = df[col].unique()
        print(f"   {col}: {len(unique_vals)} unique — {list(unique_vals)}")


if __name__ == "__main__":
    main()
