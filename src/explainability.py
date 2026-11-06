"""
Explainability Module (RBI Compliance)
======================================
SHAP-based local and global model interpretability tools to generate
explainable reason codes required by regulatory authorities (e.g., RBI).
"""

import pandas as pd
import numpy as np
import shap


def compute_shap_explanations(model, X_data: pd.DataFrame):
    """Compute TreeExplainer SHAP values for tree-based models."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_data)
    return explainer, shap_values


def generate_applicant_reason_codes(shap_values_array: np.ndarray,
                                   feature_names: list,
                                   X_df: pd.DataFrame,
                                   probabilities: np.ndarray,
                                   top_n: int = 3) -> pd.DataFrame:
    """
    Generate adverse action notice / denial reason codes for credit risk underwriter compliance.
    Extracts the top negative contributors to an applicant's default risk score.
    """
    records = []
    for idx in range(len(X_df)):
        sample_shap = shap_values_array[idx]
        # Sort descending by positive SHAP contribution (features driving default probability up)
        sorted_indices = np.argsort(-sample_shap)

        adverse_reasons = []
        for feat_idx in sorted_indices[:top_n]:
            feat_name = feature_names[feat_idx]
            impact = sample_shap[feat_idx]
            val = X_df.iloc[idx, feat_idx]
            if impact > 0:
                adverse_reasons.append(f"{feat_name} (+{impact:.3f} log-odds)")

        p_default = float(probabilities[idx])
        records.append({
            'applicant_id': int(X_df.index[idx]),
            'default_probability': round(p_default, 4),
            'risk_score': round(p_default * 100, 1),
            'primary_reason': adverse_reasons[0] if len(adverse_reasons) > 0 else 'N/A',
            'secondary_reason': adverse_reasons[1] if len(adverse_reasons) > 1 else 'N/A',
            'tertiary_reason': adverse_reasons[2] if len(adverse_reasons) > 2 else 'N/A',
        })

    return pd.DataFrame(records)
