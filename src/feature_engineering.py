"""
Feature Engineering Module
==========================
Reusable utilities for encoding, feature creation, scaling, and data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


# ── Loan grade ordinal mapping ──
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
DEFAULT_MAP = {'N': 0, 'Y': 1}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables:
    - loan_grade: ordinal (A=1 ... G=7)
    - cb_person_default_on_file: binary (N=0, Y=1)
    - person_home_ownership, loan_intent: one-hot encoded
    """
    df = df.copy()

    # Ordinal encode loan grade
    df['loan_grade'] = df['loan_grade'].map(GRADE_MAP)

    # Binary encode default on file
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(DEFAULT_MAP)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True, dtype=int)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing ones:
    - income_to_loan_ratio: ability to repay (higher = safer)
    - emp_to_age_ratio: employment stability relative to age
    - credit_utilization_proxy: combined debt burden signal
    """
    df = df.copy()

    # Income to loan ratio — higher means more capacity to repay
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']

    # Employment stability — ratio of employment length to age
    df['emp_to_age_ratio'] = df['person_emp_length'] / df['person_age']

    # Credit utilization proxy — combination of loan burden and interest rate
    df['credit_utilization_proxy'] = df['loan_percent_income'] * df['loan_int_rate']

    return df


def prepare_data(df: pd.DataFrame, target_col: str = 'loan_status', test_size: float = 0.2, random_state: int = 42):
    """
    Full preparation pipeline:
    1. Encode features
    2. Create engineered features
    3. Stratified train/test split
    4. Scale numeric features

    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Encode and engineer
    df = encode_features(df)
    df = create_features(df)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    print(f"✅ Data prepared: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test samples")
    print(f"   Features: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Apply SMOTE to balance the training set."""
    print(f"\n⚖️  Class distribution BEFORE SMOTE:")
    print(f"   Non-Default: {(y_train == 0).sum():,} | Default: {(y_train == 1).sum():,}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"   Class distribution AFTER SMOTE:")
    print(f"   Non-Default: {(y_resampled == 0).sum():,} | Default: {(y_resampled == 1).sum():,}")

    return X_resampled, y_resampled
