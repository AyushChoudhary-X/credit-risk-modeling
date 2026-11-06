"""
Data Preprocessing Module
=========================
Reusable utilities for loading, cleaning, and validating the credit risk dataset.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from the specified filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data quality report."""
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
    }
    
    print(f"\n📊 Dataset Shape: {report['shape']}")
    print(f"🔄 Duplicate Rows: {report['duplicates']}")
    print(f"\n❌ Missing Values:")
    for col, count in report['missing_values'].items():
        if count > 0:
            pct = report['missing_pct'][col]
            print(f"   {col}: {count:,} ({pct}%)")
    
    return report


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unrealistic outliers based on domain knowledge:
    - person_age > 100 (unrealistic for borrowers)
    - person_emp_length > 60 (employment longer than working life)
    - person_income > 4,000,000 (extreme outliers in this dataset)
    """
    initial_len = len(df)
    
    # Age outliers
    df = df[df['person_age'] <= 100].copy()
    
    # Employment length outliers
    df = df[df['person_emp_length'] <= 60].copy()
    
    # Extreme income outliers (beyond 99.9th percentile)
    income_cap = df['person_income'].quantile(0.999)
    df = df[df['person_income'] <= income_cap].copy()
    
    removed = initial_len - len(df)
    print(f"🗑️  Removed {removed:,} outlier rows ({removed/initial_len*100:.2f}%)")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - person_emp_length: fill with median (skewed distribution)
    - loan_int_rate: fill with median grouped by loan_grade (interest rates correlate with grade)
    """
    # Employment length — fill with median
    if df['person_emp_length'].isnull().sum() > 0:
        median_emp = df['person_emp_length'].median()
        df['person_emp_length'] = df['person_emp_length'].fillna(median_emp)
        print(f"   ✅ person_emp_length: filled {df['person_emp_length'].isnull().sum()} remaining NaN with median ({median_emp})")
    
    # Interest rate — fill with grade-wise median (more accurate)
    if df['loan_int_rate'].isnull().sum() > 0:
        grade_medians = df.groupby('loan_grade')['loan_int_rate'].median()
        for grade, median_rate in grade_medians.items():
            mask = (df['loan_int_rate'].isnull()) & (df['loan_grade'] == grade)
            df.loc[mask, 'loan_int_rate'] = median_rate
        
        # If any still remain (edge case), fill with overall median
        if df['loan_int_rate'].isnull().sum() > 0:
            df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
        
        print(f"   ✅ loan_int_rate: filled using grade-wise median imputation")
    
    remaining_nulls = df.isnull().sum().sum()
    print(f"\n📋 Remaining null values: {remaining_nulls}")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Run validation checks on the cleaned dataset."""
    checks = []
    
    # No nulls
    null_check = df.isnull().sum().sum() == 0
    checks.append(("No missing values", null_check))
    
    # Target variable is binary
    target_check = set(df['loan_status'].unique()) == {0, 1}
    checks.append(("Target is binary (0/1)", target_check))
    
    # Age range is reasonable
    age_check = (df['person_age'].min() >= 18) and (df['person_age'].max() <= 100)
    checks.append(("Age range 18-100", age_check))
    
    # Employment length is reasonable
    emp_check = df['person_emp_length'].max() <= 60
    checks.append(("Employment length ≤ 60", emp_check))
    
    # Loan amount is positive
    loan_check = (df['loan_amnt'] > 0).all()
    checks.append(("Loan amount > 0", loan_check))
    
    print("\n✅ Data Validation Results:")
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def clean_data(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Full cleaning pipeline: load → inspect → remove outliers → handle missing → validate → save.
    """
    print("=" * 60)
    print("🔧 CREDIT RISK DATA CLEANING PIPELINE")
    print("=" * 60)
    
    # Load
    df = load_raw_data(raw_path)
    
    # Inspect
    inspect_data(df)
    
    # Clean
    print("\n--- Removing Outliers ---")
    df = remove_outliers(df)
    
    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)
    
    # Validate
    validate_data(df)
    
    # Save
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n💾 Cleaned data saved to: {processed_path}")
    print(f"   Final shape: {df.shape}")
    
    return df
