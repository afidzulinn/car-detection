"""
P0-04, P0-08, P2-07:
Unified Decision Threshold Policy, Class Confidence Semantics, and Calibration Metrics.

Rules enforced:
  1. Single decision rule everywhere: pred_open = int(prob_open >= threshold).
  2. Predicted class confidence:
       confidence_predicted_class = prob_open if pred_open == 1 else (1.0 - prob_open)
     Never report P(OPEN)=0.01 as "1% confidence" for a CLOSED prediction.
  3. Threshold calibration is allowed ONLY on validation data, frozen with metadata,
     and applied once to independent test sets without re-tuning.
"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score


def predict_from_prob(
    prob_open: Union[float, np.ndarray],
    threshold: Union[float, Sequence[float], np.ndarray] = 0.50,
) -> Union[int, np.ndarray]:
    """
    Canonical binary decision rule: OPEN (1) iff prob_open >= threshold.
    """
    if isinstance(prob_open, np.ndarray):
        thr = np.asarray(threshold, dtype=np.float64)
        return (prob_open >= thr).astype(int)
    return int(float(prob_open) >= float(threshold))


def confidence_of_predicted_class(
    prob_open: Union[float, np.ndarray],
    pred_open: Union[int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Return the confidence in the predicted class:
      - If pred_open == 1 (OPEN):   confidence = P(OPEN)
      - If pred_open == 0 (CLOSED): confidence = P(CLOSED) = 1 - P(OPEN)
    """
    if isinstance(prob_open, np.ndarray):
        pred_arr = np.asarray(pred_open, dtype=int)
        return np.where(pred_arr == 1, prob_open, 1.0 - prob_open)
    return float(prob_open) if int(pred_open) == 1 else float(1.0 - float(prob_open))


def find_optimal_threshold_on_val(
    val_probs_1d: np.ndarray,
    val_targets_1d: np.ndarray,
    grid_min: float = 0.20,
    grid_max: float = 0.80,
    grid_step: float = 0.02,
) -> float:
    """
    Select the threshold in [grid_min, grid_max] that maximizes F1 on the validation split.
    Ties are broken by choosing the threshold closest to 0.50 for maximum margin stability.
    """
    best_f1 = -1.0
    best_thr = 0.50
    for thr in np.arange(grid_min, grid_max + 1e-9, grid_step):
        thr_r = round(float(thr), 2)
        preds = predict_from_prob(val_probs_1d, thr_r)
        f1 = float(f1_score(val_targets_1d, preds, zero_division=0))
        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and abs(thr_r - 0.50) < abs(best_thr - 0.50)
        ):
            best_f1 = f1
            best_thr = thr_r
    return round(float(best_thr), 2)


def calibrate_thresholds_on_val(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """
    Calibrate per-label decision thresholds strictly on the validation split (N, K).
    """
    thresholds: Dict[str, float] = {}
    for i, label in enumerate(labels):
        thresholds[label] = find_optimal_threshold_on_val(val_probs[:, i], val_targets[:, i])
    return thresholds


def compute_ece(probs_1d: np.ndarray, targets_1d: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary probabilities.
    """
    preds = (probs_1d >= 0.50).astype(int)
    confs = confidence_of_predicted_class(probs_1d, preds)
    accs = (preds == targets_1d).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs_1d)
    if n == 0:
        return 0.0

    for b_idx in range(n_bins):
        lo, hi = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confs >= lo) & (confs <= hi)
        else:
            in_bin = (confs > lo) & (confs <= hi)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accs[in_bin])
            conf_in_bin = np.mean(confs[in_bin])
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def compute_calibration_summary(
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
) -> dict:
    """
    Compute per-label and macro Expected Calibration Error (ECE) and Brier Score.
    """
    per_label = {}
    eces = []
    briers = []
    for i, label in enumerate(labels):
        e = compute_ece(probs[:, i], targets[:, i])
        b = float(brier_score_loss(targets[:, i], probs[:, i]))
        per_label[label] = {"ece": round(e, 5), "brier_score": round(b, 5)}
        eces.append(e)
        briers.append(b)
    return {
        "macro_ece": round(float(np.mean(eces)), 5),
        "macro_brier_score": round(float(np.mean(briers)), 5),
        "per_label": per_label,
    }
