"""
Profit Optimizer & Cost-Sensitive Analysis Module
=================================================
Aligns statistical classification with banking economics using custom cost matrices
and profit-maximizing decision threshold tuning.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class CreditCostMatrix:
    """
    Cost Matrix capturing commercial banking economics:
    - FP: Cost of granting loan to defaulter (Loss = full principal, e.g. ₹5,00,000)
    - FN: Cost of rejecting good borrower (Opportunity cost = lost interest, e.g. ₹50,000)
    - TN: Profit from good loan (Net interest margin earned, e.g. ₹75,000)
    - TP: Defaulter correctly flagged & rejected (Risk avoided, cost = 0)
    """

    def __init__(self, cost_fp: float = 500000.0, cost_fn: float = 50000.0, profit_tn: float = 75000.0, profit_tp: float = 0.0):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.profit_tn = profit_tn
        self.profit_tp = profit_tp

    def compute_profit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate net profit from confusion matrix outcomes."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        net_profit = (tn * self.profit_tn) - (fp * self.cost_fp) - (fn * self.cost_fn) + (tp * self.profit_tp)
        return {
            'net_profit': float(net_profit),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }

    def optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, steps: int = 100):
        """Sweep thresholds from 0.01 to 0.99 to find profit-maximizing threshold."""
        thresholds = np.linspace(0.01, 0.99, steps)
        records = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            res = self.compute_profit(y_true, y_pred)
            res['threshold'] = float(t)
            res['accuracy'] = float(accuracy_score(y_true, y_pred))
            res['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            res['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            res['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            records.append(res)

        df = pd.DataFrame(records)
        optimal_idx = df['net_profit'].idxmax()
        return df, df.iloc[optimal_idx]
