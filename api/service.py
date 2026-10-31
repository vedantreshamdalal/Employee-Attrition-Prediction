"""Model loading, scoring, dataset summaries, and drift checks."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from joblib import load

from api.metrics import (
    attrition_predictions_total,
    model_available,
    prediction_requests_total,
    retention_predictions_total,
)
from ml.drift import compare_drift, load_reference
from ml.features import (
    FEATURE_COLUMNS,
    RISK_HIGH_AT,
    RISK_MEDIUM_AT,
    TARGET_COLUMN,
    comparison_path,
    dataset_path,
    drift_report_path,
    metrics_path,
    model_path,
    reference_path,
)

MODEL = None
METRICS: dict[str, Any] | None = None
COMPARISON: dict[str, Any] | None = None
REFERENCE: pd.DataFrame | None = None
LOAD_ERROR: str | None = None
REFERENCE_ERROR: str | None = None


class ModelNotAvailable(RuntimeError):
    """Raised when a request needs the saved pipeline and it is not loaded."""


def risk_category(probability_attrition: float) -> str:
    if probability_attrition >= RISK_HIGH_AT:
        return "High"
    if probability_attrition >= RISK_MEDIUM_AT:
        return "Medium"
    return "Low"


def load_artifacts() -> None:
    """Load the pipeline, evaluation files, and training reference sample."""
    global MODEL, METRICS, COMPARISON, REFERENCE, LOAD_ERROR, REFERENCE_ERROR
    try:
        MODEL = load(model_path())
        METRICS = json.loads(metrics_path().read_text(encoding="utf-8"))
        COMPARISON = json.loads(comparison_path().read_text(encoding="utf-8"))
        LOAD_ERROR = None
        model_available.set(1)
    except Exception as exc:
        MODEL = None
        METRICS = None
        COMPARISON = None
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        model_available.set(0)
    try:
        REFERENCE = load_reference(reference_path())
        REFERENCE_ERROR = None
    except Exception as exc:
        REFERENCE = None
        REFERENCE_ERROR = f"{type(exc).__name__}: {exc}"


def model_is_loaded() -> bool:
    return MODEL is not None


def load_error_message() -> str | None:
    return LOAD_ERROR


def require_model():
    if MODEL is None:
        raise ModelNotAvailable(LOAD_ERROR or "The model pipeline is not loaded. Train it with python -m ml.train.")
    return MODEL


def predict_records(records: list[dict]) -> list[dict]:
    model = require_model()
    frame = pd.DataFrame(records)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction records are missing columns: {missing}")
    frame = frame[FEATURE_COLUMNS]
    labels = model.predict(frame)
    probabilities = model.predict_proba(frame)
    classes = list(model.named_steps["model"].classes_)
    yes_index = classes.index(1)
    no_index = classes.index(0)
    predictions = []
    yes_count = 0
    no_count = 0
    for label, row in zip(labels, probabilities):
        probability_attrition = min(1.0, max(0.0, round(float(row[yes_index]), 6)))
        probability_retention = min(1.0, max(0.0, round(float(row[no_index]), 6)))
        attrition = "Yes" if int(label) == 1 else "No"
        if attrition == "Yes":
            yes_count += 1
        else:
            no_count += 1
        predictions.append(
            {
                "attrition": attrition,
                "probability_attrition": probability_attrition,
                "probability_retention": probability_retention,
                "risk_category": risk_category(probability_attrition),
                "risk_thresholds": {"medium_at": RISK_MEDIUM_AT, "high_at": RISK_HIGH_AT},
            }
        )
    prediction_requests_total.inc(len(predictions))
    if yes_count:
        attrition_predictions_total.inc(yes_count)
    if no_count:
        retention_predictions_total.inc(no_count)
    return predictions


def dataset_overview() -> dict:
    frame = pd.read_csv(dataset_path())
    total = int(len(frame))
    left = int((frame[TARGET_COLUMN] == "Yes").sum())
    stayed = total - left

    def grouped(column: str) -> list[dict]:
        rows = []
        for name, group in frame.groupby(column):
            group_left = int((group[TARGET_COLUMN] == "Yes").sum())
            group_total = int(len(group))
            rows.append(
                {
                    "name": str(name),
                    "total": group_total,
                    "left": group_left,
                    "stayed": group_total - group_left,
                    "attrition_percentage": round(100 * group_left / group_total, 2) if group_total else 0.0,
                }
            )
        rows.sort(key=lambda item: item["attrition_percentage"], reverse=True)
        return rows

    def income_or_years(column: str) -> dict:
        left_values = frame.loc[frame[TARGET_COLUMN] == "Yes", column].astype(int)
        stayed_values = frame.loc[frame[TARGET_COLUMN] == "No", column].astype(int)
        return {
            "left": left_values.tolist(),
            "stayed": stayed_values.tolist(),
            "left_mean": round(float(left_values.mean()), 2),
            "stayed_mean": round(float(stayed_values.mean()), 2),
            "left_median": round(float(left_values.median()), 2),
            "stayed_median": round(float(stayed_values.median()), 2),
        }

    return {
        "total_employees": total,
        "employees_left": left,
        "employees_stayed": stayed,
        "attrition_percentage": round(100 * left / total, 2) if total else 0.0,
        "attrition_distribution": {"Yes": left, "No": stayed},
        "department_attrition": grouped("Department"),
        "job_role_attrition": grouped("JobRole"),
        "monthly_income": income_or_years("MonthlyIncome"),
        "years_at_company": income_or_years("YearsAtCompany"),
        "source": dataset_path().name,
    }


def performance_payload() -> dict:
    if METRICS is None or COMPARISON is None:
        raise ModelNotAvailable(LOAD_ERROR or "Model metrics are not available. Train the pipeline first.")
    return {
        "best_model": METRICS["best_model"],
        "selection_reason": METRICS["selection_reason"],
        "positive_class": METRICS["positive_class"],
        "metrics": METRICS["metrics"],
        "confusion_matrix": METRICS["confusion_matrix"],
        "classification_report": METRICS["classification_report"],
        "feature_importance_source": METRICS["feature_importance_source"],
        "feature_importance_note": METRICS["feature_importance_note"],
        "feature_importance_top5": METRICS["feature_importance_top5"],
        "feature_importance": METRICS["feature_importance"],
        "comparison": COMPARISON,
        "split": METRICS["split"],
        "class_balance": METRICS["class_balance"],
        "model_card": METRICS["model_card"],
        "dropped_columns": METRICS["dropped_columns"],
    }


def run_drift(records: list[dict]) -> dict:
    if REFERENCE is None:
        raise ModelNotAvailable(
            REFERENCE_ERROR or "Training reference features are missing. Run python -m ml.train."
        )
    report = compare_drift(REFERENCE, pd.DataFrame(records))
    destination = drift_report_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def latest_drift() -> dict:
    path = drift_report_path()
    if not path.is_file():
        return {
            "status": "not_checked",
            "drift_type": "data_drift",
            "message": "No drift check has been run yet. Upload employee records to compare them with training data.",
            "disclaimer": (
                "Data drift describes input distributions. It is not a measure of prediction accuracy."
            ),
        }
    return json.loads(path.read_text(encoding="utf-8"))
