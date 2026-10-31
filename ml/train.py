"""Train attrition classifiers and save the selected pipeline.

The saved model is the pipeline fitted on the training split only, so the
metrics in metrics.json describe that same model. Precision, recall, and F1
refer to the minority class Attrition = Yes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    comparison_path,
    dataset_path,
    eda_path,
    metrics_path,
    model_path,
    reference_path,
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERICAL_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or dataset_path()
    frame = pd.read_csv(csv_path)
    required = [TARGET_COLUMN, *FEATURE_COLUMNS, *DROP_COLUMNS]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset has no rows.")
    unexpected = set(frame[TARGET_COLUMN].dropna().unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Attrition contains unexpected values: {sorted(unexpected)}")
    return frame


def exploratory_summary(frame: pd.DataFrame) -> dict:
    missing_by_column = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if int(count) > 0
    }
    attrition_counts = frame[TARGET_COLUMN].value_counts().to_dict()
    total = int(len(frame))
    yes_count = int(attrition_counts.get("Yes", 0))
    numeric_summary = {}
    for column in NUMERICAL_FEATURES:
        series = pd.to_numeric(frame[column], errors="coerce")
        numeric_summary[column] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
        }
    grouped_rates = {}
    for column in ("Department", "JobRole", "OverTime", "BusinessTravel", "Gender", "MaritalStatus"):
        rows = []
        for name, group in frame.groupby(column):
            left = int((group[TARGET_COLUMN] == "Yes").sum())
            group_total = int(len(group))
            rows.append(
                {
                    "name": str(name),
                    "total": group_total,
                    "left": left,
                    "attrition_percentage": round(100 * left / group_total, 2),
                }
            )
        rows.sort(key=lambda item: item["attrition_percentage"], reverse=True)
        grouped_rates[column] = rows
    return {
        "rows": total,
        "columns": int(frame.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "attrition_counts": {str(key): int(value) for key, value in attrition_counts.items()},
        "attrition_percentage": round(100 * yes_count / total, 4),
        "removed_columns": DROP_COLUMNS,
        "numerical_summary": numeric_summary,
        "attrition_rate_by_category": grouped_rates,
    }


def _evaluate(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    classes = list(pipeline.named_steps["model"].classes_)
    yes_index = classes.index(1)
    scores = probabilities[:, yes_index]
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )
    attrition_precision = precision_score(y_test, predicted, pos_label=1, zero_division=0)
    attrition_recall = recall_score(y_test, predicted, pos_label=1, zero_division=0)
    attrition_f1 = f1_score(y_test, predicted, pos_label=1, zero_division=0)
    retention_precision = precision_score(y_test, predicted, pos_label=0, zero_division=0)
    retention_recall = recall_score(y_test, predicted, pos_label=0, zero_division=0)
    retention_f1 = f1_score(y_test, predicted, pos_label=0, zero_division=0)
    return {
        "name": name,
        "accuracy": round(float(accuracy_score(y_test, predicted)), 6),
        "precision": round(float(attrition_precision), 6),
        "recall": round(float(attrition_recall), 6),
        "f1_score": round(float(attrition_f1), 6),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 6),
        "precision_attrition": round(float(attrition_precision), 6),
        "recall_attrition": round(float(attrition_recall), 6),
        "f1_attrition": round(float(attrition_f1), 6),
        "precision_retention": round(float(retention_precision), 6),
        "recall_retention": round(float(retention_recall), 6),
        "f1_retention": round(float(retention_f1), 6),
        "confusion_matrix": {
            "labels": ["No", "Yes"],
            "rows": "actual",
            "columns": "predicted",
            "matrix": matrix,
        },
        "classification_report": _json_ready(report),
    }


def _original_feature(encoded_name: str) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.split("__")[-1]
    tail = encoded_name.split("cat__", 1)[-1]
    for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if f"{column}_" in tail:
            return column
    return tail


def feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    importances = pipeline.named_steps["model"].feature_importances_
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if len(importances) != len(names):
        raise RuntimeError(
            "Feature importance length does not match the preprocessing output. "
            f"importances={len(importances)} names={len(names)}"
        )
    totals: dict[str, float] = {column: 0.0 for column in FEATURE_COLUMNS}
    for name, score in zip(names, importances):
        feature = _original_feature(str(name))
        if feature not in totals:
            raise RuntimeError(
                "Could not map an encoded feature back to a training column. "
                f"encoded={name!r} mapped={feature!r} sample={list(map(str, names[:12]))}"
            )
        totals[feature] += float(score)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": round(score, 6)} for feature, score in ranked]


def train_models(data: Path | None = None) -> dict:
    frame = load_dataset(data)
    summary = exploratory_summary(frame)
    features = frame[FEATURE_COLUMNS].copy()
    target = (frame[TARGET_COLUMN] == "Yes").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    models = build_models()
    fitted: dict[str, Pipeline] = {}
    evaluations = []
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        evaluations.append(_evaluate(name, pipeline, x_test, y_test))

    def rank_key(item: dict) -> tuple[float, float, float]:
        return (item["f1_attrition"], item["recall_attrition"], item["roc_auc"])

    best = max(evaluations, key=rank_key)
    others = [item for item in evaluations if item["name"] != best["name"]]
    other_text = " ".join(
        (
            f"{item['name']} has Attrition F1 {item['f1_attrition']:.4f}, "
            f"Attrition recall {item['recall_attrition']:.4f}, and ROC-AUC {item['roc_auc']:.4f}."
        )
        for item in others
    )
    selection_reason = (
        f"Selected {best['name']} because it has the best minority-class performance. "
        f"Primary metric is F1 on Attrition=Yes ({best['f1_attrition']:.4f}), "
        f"with Attrition recall ({best['recall_attrition']:.4f}) and ROC-AUC "
        f"({best['roc_auc']:.4f}) as tie-breakers. Accuracy was not the selection metric. "
        f"{other_text}"
    ).strip()

    importances = feature_importance(fitted["random_forest"])
    importance_total = sum(item["importance"] for item in importances)
    if abs(importance_total - 1) > 0.02:
        raise RuntimeError(f"Aggregated feature importances sum to {importance_total}, expected about 1.")

    destination = model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump(fitted[best["name"]], destination, compress=3)
    x_train.to_csv(reference_path(), index=False)

    metrics = {
        "best_model": best["name"],
        "selection_metric": "f1_attrition",
        "selection_tie_breakers": ["recall_attrition", "roc_auc"],
        "selection_reason": selection_reason,
        "positive_class": "Yes",
        "random_state": RANDOM_STATE,
        "model_card": {
            "algorithm": best["name"],
            "target": TARGET_COLUMN,
            "positive_class": "Yes",
            "class_weight": "balanced",
            "preprocessing": [
                "median imputation for numerical features",
                "most-frequent imputation for categorical features",
                "standard scaling for numerical features",
                "one-hot encoding for categorical features",
            ],
            "random_state": RANDOM_STATE,
            "fit_on": "training split only",
        },
        "split": {
            "test_size": TEST_SIZE,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_attrition_yes": int(y_train.sum()),
            "test_attrition_yes": int(y_test.sum()),
            "train_attrition_no": int((y_train == 0).sum()),
            "test_attrition_no": int((y_test == 0).sum()),
        },
        "class_balance": {
            "Yes": int(summary["attrition_counts"].get("Yes", 0)),
            "No": int(summary["attrition_counts"].get("No", 0)),
            "attrition_percentage": summary["attrition_percentage"],
        },
        "metrics": {
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_score": best["f1_score"],
            "roc_auc": best["roc_auc"],
            "precision_attrition": best["precision_attrition"],
            "recall_attrition": best["recall_attrition"],
            "f1_attrition": best["f1_attrition"],
            "precision_retention": best["precision_retention"],
            "recall_retention": best["recall_retention"],
            "f1_retention": best["f1_retention"],
        },
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
        "feature_importance_source": "random_forest_impurity",
        "feature_importance_note": (
            "Importances are global Random Forest impurity scores, summed across "
            "one-hot columns back to the original feature. They are not causal effects "
            "and they are not an explanation of one employee."
        ),
        "feature_importance_top5": importances[:5],
        "feature_importance": importances,
        "dropped_columns": DROP_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "eda_summary_file": str(eda_path().relative_to(eda_path().parents[1])),
    }
    comparison = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "positive_class": "Yes",
        "selection_metric": "f1_attrition",
        "tie_breakers": ["recall_attrition", "roc_auc"],
        "best_model": best["name"],
        "selection_reason": selection_reason,
        "models": evaluations,
    }
    metrics_path().write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison_path().write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    eda_path().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rows: {summary['rows']}  Missing values: {summary['missing_values']}")
    print(
        "Attrition: "
        f"Yes={summary['attrition_counts'].get('Yes', 0)} "
        f"No={summary['attrition_counts'].get('No', 0)} "
        f"({summary['attrition_percentage']:.2f}%)"
    )
    print(f"Train rows: {len(x_train)}  Test rows: {len(x_test)}")
    print("Precision, recall, and F1 below are for Attrition = Yes.")
    for item in evaluations:
        print(
            f"{item['name']:22} accuracy={item['accuracy']:.4f} "
            f"precision={item['precision']:.4f} recall={item['recall']:.4f} "
            f"f1={item['f1_score']:.4f} roc_auc={item['roc_auc']:.4f}"
        )
    print(f"Best model: {best['name']}")
    print("Top features:")
    for item in importances[:5]:
        print(f"  {item['feature']:28} {item['importance']:.4f}")
    print(f"Saved model to {destination}")
    return {
        "best_model": best["name"],
        "model_path": str(destination),
        "metrics_path": str(metrics_path()),
        "comparison_path": str(comparison_path()),
        "reference_path": str(reference_path()),
        "eda_path": str(eda_path()),
    }


def main() -> None:
    try:
        train_models()
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
