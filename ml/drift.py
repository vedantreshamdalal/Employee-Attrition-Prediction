"""Compare incoming employee records with the training-feature distribution.

This module reports data drift only. It does not estimate accuracy, and it
cannot show model performance degradation unless labeled outcomes are evaluated
separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERICAL_FEATURES

MIN_ROWS = 30
KS_PVALUE = 0.05
MEAN_SHIFT_DRIFT = 0.25
MEAN_SHIFT_WARNING = 0.10
TVD_DRIFT = 0.20
TVD_WARNING = 0.10
CHI2_PVALUE = 0.05

DRIFT_DISCLAIMER = (
    "This report measures data drift: changes in the distribution of model inputs "
    "compared with the training set. It does not measure model performance degradation. "
    "Performance evaluation requires ground-truth attrition labels, which unlabeled "
    "prediction requests do not provide."
)


def load_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference data is missing columns: {missing}")
    return _coerce(frame[FEATURE_COLUMNS].copy())


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in NUMERICAL_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].astype("string")
    return prepared


def _round(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 6)


def _numerical_status(reference: pd.Series, current: pd.Series) -> dict:
    statistic, p_value = ks_2samp(reference.dropna(), current.dropna())
    reference_std = float(reference.std(ddof=0))
    scale = reference_std if reference_std > 0 else 1.0
    mean_shift = abs(float(current.mean()) - float(reference.mean())) / scale
    if p_value < KS_PVALUE and mean_shift >= MEAN_SHIFT_DRIFT:
        status = "drift"
    elif p_value < KS_PVALUE and mean_shift >= MEAN_SHIFT_WARNING:
        status = "warning"
    else:
        status = "stable"
    return {
        "feature": reference.name,
        "type": "numerical",
        "status": status,
        "statistic_name": "ks_statistic",
        "statistic": _round(float(statistic)),
        "p_value": _round(float(p_value)),
        "mean_shift_std": _round(mean_shift),
    }


def _categorical_status(reference: pd.Series, current: pd.Series) -> dict:
    ref_counts = reference.fillna("Missing").astype(str).value_counts()
    cur_counts = current.fillna("Missing").astype(str).value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_arr = np.array([ref_counts.get(category, 0) for category in categories], dtype=float)
    cur_arr = np.array([cur_counts.get(category, 0) for category in categories], dtype=float)
    ref_share = ref_arr / ref_arr.sum()
    cur_share = cur_arr / cur_arr.sum()
    tvd = float(0.5 * np.abs(ref_share - cur_share).sum())
    table = np.vstack([ref_arr, cur_arr])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    p_value = None
    if table.shape[1] >= 2 and table.sum() > 0 and (table.sum(axis=1) > 0).all():
        _chi2, p_value, _dof, _expected = chi2_contingency(table)
        p_value = float(p_value)
    large_gap = tvd >= TVD_DRIFT
    significant = p_value is None or p_value < CHI2_PVALUE
    if large_gap and significant:
        status = "drift"
    elif tvd >= TVD_WARNING:
        status = "warning"
    else:
        status = "stable"
    return {
        "feature": reference.name,
        "type": "categorical",
        "status": status,
        "statistic_name": "total_variation_distance",
        "statistic": _round(tvd),
        "p_value": _round(p_value),
        "mean_shift_std": None,
    }


def compare_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    missing = [column for column in FEATURE_COLUMNS if column not in current.columns]
    if missing:
        raise ValueError(f"Incoming records are missing columns: {missing}")
    reference = _coerce(reference)
    current = _coerce(current)
    current = current[FEATURE_COLUMNS]
    dropped = int(current[FEATURE_COLUMNS].isna().any(axis=1).sum())
    current = current.dropna(subset=FEATURE_COLUMNS)
    base = {
        "drift_type": "data_drift",
        "compared_with": "training_split_features",
        "reference_rows": int(len(reference)),
        "incoming_rows": int(len(current)),
        "rows_dropped_for_missing_values": dropped,
        "disclaimer": DRIFT_DISCLAIMER,
        "thresholds": {
            "minimum_rows": MIN_ROWS,
            "numerical_ks_pvalue": KS_PVALUE,
            "numerical_mean_shift_std_drift": MEAN_SHIFT_DRIFT,
            "numerical_mean_shift_std_warning": MEAN_SHIFT_WARNING,
            "categorical_tvd_drift": TVD_DRIFT,
            "categorical_tvd_warning": TVD_WARNING,
            "categorical_chi2_pvalue": CHI2_PVALUE,
        },
    }
    if len(current) < MIN_ROWS:
        base.update(
            {
                "status": "insufficient_data",
                "message": (
                    f"At least {MIN_ROWS} complete rows are required to compare distributions. "
                    f"Received {len(current)}."
                ),
                "features": [],
                "drifted_features": [],
            }
        )
        return base

    features = []
    for column in NUMERICAL_FEATURES:
        reference[column].name = column
        current[column].name = column
        features.append(_numerical_status(reference[column], current[column]))
    for column in CATEGORICAL_FEATURES:
        reference[column].name = column
        current[column].name = column
        features.append(_categorical_status(reference[column], current[column]))
    drifted = [item["feature"] for item in features if item["status"] == "drift"]
    warnings = [item["feature"] for item in features if item["status"] == "warning"]
    if drifted:
        status = "drift_detected"
        message = "One or more input features differ from the training distribution."
    elif warnings:
        status = "warning"
        message = "Some input features show a smaller distribution change."
    else:
        status = "no_drift"
        message = "No monitored input feature crossed the drift thresholds."
    base.update(
        {
            "status": status,
            "message": message,
            "features": features,
            "drifted_features": drifted,
            "warning_features": warnings,
        }
    )
    return base
