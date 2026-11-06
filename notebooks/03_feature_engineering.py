"""
03 — Feature Engineering Pipeline
===================================
Encode, engineer, split, and balance the credit risk dataset.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.feature_engineering import encode_features, create_features, prepare_data, apply_smote

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_data.csv')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def main():
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} records\n")

    # Show original categorical values
    print("📝 Original Categorical Values:")
    for col in ['loan_grade', 'person_home_ownership', 'loan_intent', 'cb_person_default_on_file']:
        print(f"   {col}: {df[col].unique().tolist()}")

    # Prepare data (encode + engineer + split + scale)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    print(f"\n📊 Feature List ({len(feature_names)} features):")
    for i, name in enumerate(feature_names, 1):
        print(f"   {i:2d}. {name}")

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Save processed data
    train_df = pd.DataFrame(X_train_smote, columns=feature_names)
    train_df['loan_status'] = y_train_smote.values
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'features_train.csv'), index=False)

    test_df = pd.DataFrame(X_test.values, columns=feature_names)
    test_df['loan_status'] = y_test.values
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'features_test.csv'), index=False)

    print(f"\n💾 Saved:")
    print(f"   Train (SMOTE): {train_df.shape} → data/processed/features_train.csv")
    print(f"   Test:          {test_df.shape} → data/processed/features_test.csv")


if __name__ == "__main__":
    main()
