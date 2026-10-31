"""Dataset, training, saved pipeline, and drift tests."""

import json

import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.drift import DRIFT_DISCLAIMER, compare_drift, load_reference
from ml.features import (
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    comparison_path,
    metrics_path,
    model_path,
    reference_path,
)
from ml.train import load_dataset


def test_dataset_loads_and_target_has_no_missing_values():
    frame = load_dataset()
    assert len(frame) > 1000
    assert TARGET_COLUMN in frame.columns
    assert frame[TARGET_COLUMN].isna().sum() == 0
    assert set(frame[TARGET_COLUMN].unique()) <= {"Yes", "No"}
    assert int(frame.isna().sum().sum()) == 0


def test_irrelevant_columns_are_excluded_from_the_model():
    for column in ("EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"):
        assert column in DROP_COLUMNS
        assert column not in FEATURE_COLUMNS
    assert "Attrition" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 27


def test_saved_split_is_stratified():
    metrics = json.loads(metrics_path().read_text(encoding="utf-8"))
    split = metrics["split"]
    train_rate = split["train_attrition_yes"] / split["train_rows"]
    test_rate = split["test_attrition_yes"] / split["test_rows"]
    assert abs(train_rate - test_rate) < 0.02
    assert split["train_rows"] + split["test_rows"] == metrics["class_balance"]["Yes"] + metrics["class_balance"]["No"]


def test_best_model_was_selected_on_minority_class_f1():
    comparison = json.loads(comparison_path().read_text(encoding="utf-8"))

    def rank(item):
        return (item["f1_attrition"], item["recall_attrition"], item["roc_auc"])

    winner = max(comparison["models"], key=rank)
    assert comparison["best_model"] == winner["name"]
    assert "accuracy" in comparison["selection_reason"].lower() or "Accuracy" in comparison["selection_reason"]
    assert comparison["selection_metric"] == "f1_attrition"


def test_saved_metrics_match_the_holdout_predictions():
    frame = load_dataset()
    features = frame[FEATURE_COLUMNS]
    target = (frame[TARGET_COLUMN] == "Yes").astype(int)
    _x_train, x_test, _y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = load(model_path())
    predicted = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, list(model.named_steps["model"].classes_).index(1)]
    saved = json.loads(metrics_path().read_text(encoding="utf-8"))["metrics"]
    assert abs(saved["roc_auc"] - roc_auc_score(y_test, scores)) < 1e-5
    assert abs(saved["f1_score"] - f1_score(y_test, predicted, pos_label=1)) < 1e-5
    assert abs(saved["accuracy"] - accuracy_score(y_test, predicted)) < 1e-5
    assert model.named_steps["model"].class_weight == "balanced"


def test_prediction_probabilities_sum_to_one():
    frame = load_dataset()
    sample = frame[FEATURE_COLUMNS].head(15)
    model = load(model_path())
    probabilities = model.predict_proba(sample)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.min() >= 0
    assert probabilities.max() <= 1
    labels = model.predict(sample)
    assert set(np.unique(labels)).issubset({0, 1})


def test_feature_importance_top_five_is_ranked():
    metrics = json.loads(metrics_path().read_text(encoding="utf-8"))
    top5 = metrics["feature_importance_top5"]
    assert len(top5) == 5
    scores = [item["importance"] for item in top5]
    assert scores == sorted(scores, reverse=True)
    assert metrics["feature_importance_source"] == "random_forest_impurity"
    assert all(item["feature"] in FEATURE_COLUMNS for item in top5)
    total = sum(item["importance"] for item in metrics["feature_importance"])
    assert abs(total - 1) < 0.02


def test_identical_reference_has_no_data_drift():
    reference = load_reference(reference_path())
    report = compare_drift(reference, reference.copy())
    assert report["status"] == "no_drift"
    assert report["drift_type"] == "data_drift"
    assert "does not measure model performance degradation" in report["disclaimer"]
    assert "accuracy" not in report


def test_income_shift_is_detected_as_data_drift():
    reference = load_reference(reference_path())
    shifted = reference.copy()
    shifted["MonthlyIncome"] = shifted["MonthlyIncome"] + 15000
    report = compare_drift(reference, shifted)
    assert report["status"] == "drift_detected"
    income = next(item for item in report["features"] if item["feature"] == "MonthlyIncome")
    assert income["status"] == "drift"
    assert "accuracy" not in report
    assert DRIFT_DISCLAIMER in report["disclaimer"]


def test_small_drift_sample_is_not_called_performance_degradation():
    reference = load_reference(reference_path())
    report = compare_drift(reference, reference.head(5).copy())
    assert report["status"] == "insufficient_data"
    assert "performance degradation" in report["disclaimer"]


def test_drift_module_rejects_missing_columns():
    reference = load_reference(reference_path())
    with pytest.raises(ValueError):
        compare_drift(reference, reference.drop(columns=["Age"]))
