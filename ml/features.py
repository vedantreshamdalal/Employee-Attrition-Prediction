"""Shared feature contract for training and inference.

DailyRate, HourlyRate, and MonthlyRate are excluded so the saved pipeline
matches the employee form. EmployeeNumber, EmployeeCount, Over18, and
StandardHours are identifiers or constants and are not used as features.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Attrition"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

DATASET_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "DailyRate",
    "HourlyRate",
    "MonthlyRate",
]

CATEGORICAL_OPTIONS: dict[str, tuple[str, ...]] = {
    "BusinessTravel": ("Non-Travel", "Travel_Frequently", "Travel_Rarely"),
    "Department": ("Human Resources", "Research & Development", "Sales"),
    "EducationField": (
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ),
    "Gender": ("Female", "Male"),
    "JobRole": (
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ),
    "MaritalStatus": ("Divorced", "Married", "Single"),
    "OverTime": ("No", "Yes"),
}

FEATURE_COLUMNS = [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column in CATEGORICAL_OPTIONS]
NUMERICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_OPTIONS]

RISK_HIGH_AT = 0.60
RISK_MEDIUM_AT = 0.35


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value:
        return Path(value)
    return None


def dataset_path() -> Path:
    candidates: list[Path] = []
    override = _env_path("DATA_PATH")
    if override is not None:
        candidates.append(override)
    candidates.append(ROOT / "data" / DATASET_FILENAME)
    candidates.append(ROOT / DATASET_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Employee attrition CSV not found. Looked in: {searched}")


def model_path() -> Path:
    return _env_path("MODEL_PATH") or (ROOT / "ml" / "model.pkl")


def metrics_path() -> Path:
    return _env_path("METRICS_PATH") or (ROOT / "ml" / "metrics.json")


def comparison_path() -> Path:
    return _env_path("COMPARISON_PATH") or (ROOT / "ml" / "model_comparison.json")


def reference_path() -> Path:
    return _env_path("REFERENCE_PATH") or (ROOT / "ml" / "reference_features.csv")


def eda_path() -> Path:
    return ROOT / "ml" / "eda_summary.json"


def drift_report_path() -> Path:
    return _env_path("DRIFT_REPORT_PATH") or (ROOT / "ml" / "drift_report.json")
