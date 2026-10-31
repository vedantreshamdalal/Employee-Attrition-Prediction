"""CSV checks used by the Streamlit batch and drift pages."""

import importlib.util
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("attrition_dashboard", ROOT / "streamlit_app" / "app.py")
DASHBOARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DASHBOARD)


def _valid_row() -> dict:
    return {
        "Age": 36,
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "DistanceFromHome": 7,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Female",
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5000,
        "NumCompaniesWorked": 1,
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }


def test_valid_csv_row_is_accepted():
    records = DASHBOARD.prepare_upload(pd.DataFrame([_valid_row()]))
    assert len(records) == 1
    assert records[0]["OverTime"] == "No"
    assert records[0]["MonthlyIncome"] == 5000


def test_empty_csv_is_rejected():
    with pytest.raises(DASHBOARD.APIError, match="empty"):
        DASHBOARD.read_uploaded_csv(BytesIO(b""))


def test_malformed_csv_is_rejected():
    payload = b'Age,BusinessTravel\n"unterminated,Travel_Rarely\n'
    with pytest.raises(DASHBOARD.APIError, match="malformed"):
        DASHBOARD.read_uploaded_csv(BytesIO(payload))


def test_header_only_csv_is_rejected():
    payload = ",".join(DASHBOARD.MODEL_FIELDS).encode() + b"\n"
    with pytest.raises(DASHBOARD.APIError, match="no employee rows"):
        DASHBOARD.read_uploaded_csv(BytesIO(payload))


def test_missing_values_are_rejected():
    row = _valid_row()
    row["MonthlyIncome"] = None
    with pytest.raises(DASHBOARD.APIError, match="Monthly income is empty"):
        DASHBOARD.prepare_upload(pd.DataFrame([row]))


def test_invalid_category_is_rejected():
    row = _valid_row()
    row["OverTime"] = "Sometimes"
    with pytest.raises(DASHBOARD.APIError, match="invalid value"):
        DASHBOARD.prepare_upload(pd.DataFrame([row]))


def test_non_numeric_value_is_rejected():
    row = _valid_row()
    row["Age"] = "thirty"
    with pytest.raises(DASHBOARD.APIError, match="not a number"):
        DASHBOARD.prepare_upload(pd.DataFrame([row]))
