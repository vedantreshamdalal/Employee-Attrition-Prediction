"""API health, scoring, batch CSV scoring, and Prometheus metrics."""

import json
from typing import get_args

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EmployeeFeatures
from ml.features import CATEGORICAL_OPTIONS, FEATURE_COLUMNS, dataset_path


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _records(count: int = 1) -> list[dict]:
    frame = pd.read_csv(dataset_path())
    return json.loads(frame[FEATURE_COLUMNS].head(count).to_json(orient="records"))


def test_api_categories_match_training_options():
    assert set(EmployeeFeatures.model_fields) == set(FEATURE_COLUMNS)
    for field, options in CATEGORICAL_OPTIONS.items():
        advertised = set(get_args(EmployeeFeatures.model_fields[field].annotation))
        assert advertised == set(options)


def test_every_dataset_row_matches_the_api_schema():
    records = _records(count=10**9)
    for record in records:
        EmployeeFeatures.model_validate(record)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_available"] is True


def test_predict(client):
    response = client.post("/predict", json=_records(1)[0])
    assert response.status_code == 200
    body = response.json()
    assert body["attrition"] in {"Yes", "No"}
    assert 0 <= body["probability_attrition"] <= 1
    assert 0 <= body["probability_retention"] <= 1
    assert abs(body["probability_attrition"] + body["probability_retention"] - 1) < 1e-4
    assert body["risk_category"] in {"Low", "Medium", "High"}


def test_predict_rejects_invalid_age(client):
    payload = _records(1)[0]
    payload["Age"] = 10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"employees": _records(4)})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    assert len(body["predictions"]) == 4
    assert all(item["attrition"] in {"Yes", "No"} for item in body["predictions"])


def test_csv_batch_prediction_uses_dataset_rows(client):
    frame = pd.read_csv(dataset_path()).head(6)
    payload = {"employees": json.loads(frame[FEATURE_COLUMNS].to_json(orient="records"))}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["count"] == 6


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"employees": []})
    assert response.status_code == 422


def test_prometheus_metrics(client):
    before = client.get("/metrics")
    assert before.status_code == 200
    assert "model_available" in before.text
    assert "process_cpu_seconds_total" in before.text
    assert "process_resident_memory_bytes" in before.text
    scored = client.post("/predict", json=_records(1)[0])
    assert scored.status_code == 200
    metrics = client.get("/metrics").text
    for name in (
        "http_requests_total",
        "prediction_requests_total",
        "http_request_duration_seconds",
        "failed_requests_total",
        "attrition_predictions_total",
        "retention_predictions_total",
    ):
        assert name in metrics
    invalid = client.post("/predict", json={"Age": 5})
    assert invalid.status_code == 422
    failed = client.get("/metrics").text
    assert 'failed_requests_total' in failed
    assert 'endpoint="/predict"' in failed or "endpoint=\"/predict\"" in failed


def test_model_performance_endpoint_uses_saved_metrics(client):
    response = client.get("/model/performance")
    assert response.status_code == 200
    body = response.json()
    saved = json.loads(
        __import__("pathlib").Path("ml/metrics.json").read_text(encoding="utf-8")
    )
    assert body["metrics"]["roc_auc"] == saved["metrics"]["roc_auc"]
    assert body["metrics"]["f1_score"] == saved["metrics"]["f1_score"]
    assert len(body["feature_importance_top5"]) == 5
    assert body["confusion_matrix"]["matrix"] == saved["confusion_matrix"]["matrix"]
    assert "models" in body["comparison"]


def test_overview_uses_the_dataset(client):
    response = client.get("/analytics/overview")
    assert response.status_code == 200
    body = response.json()
    frame = pd.read_csv(dataset_path())
    assert body["total_employees"] == len(frame)
    assert body["employees_left"] == int((frame["Attrition"] == "Yes").sum())
    assert body["employees_stayed"] == int((frame["Attrition"] == "No").sum())
    assert body["department_attrition"]
    assert body["job_role_attrition"]
    assert body["monthly_income"]["left_mean"] > 0
    assert body["years_at_company"]["stayed_mean"] >= 0


def test_drift_endpoint_does_not_invent_accuracy(client):
    reference = pd.read_csv("ml/reference_features.csv")
    records = json.loads(reference.head(40).to_json(orient="records"))
    response = client.post("/drift/check", json={"employees": records})
    assert response.status_code == 200
    body = response.json()
    assert body["drift_type"] == "data_drift"
    assert "accuracy" not in body
    assert "does not measure model performance degradation" in body["disclaimer"]
    latest = client.get("/drift/latest")
    assert latest.status_code == 200
    assert latest.json()["drift_type"] == "data_drift"


def test_openapi_lists_required_routes(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict/batch" in paths
    assert "/metrics" in paths
    docs = client.get("/docs")
    assert docs.status_code == 200
