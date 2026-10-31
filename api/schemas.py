"""Request and response models for the attrition API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.features import RISK_HIGH_AT, RISK_MEDIUM_AT


class EmployeeFeatures(BaseModel):
    """One employee record using the same columns the model was trained on."""

    model_config = ConfigDict(extra="ignore")

    Age: int = Field(..., ge=18, le=80, description="Age in years.")
    BusinessTravel: Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
    Department: Literal["Human Resources", "Research & Development", "Sales"]
    DistanceFromHome: int = Field(..., ge=0, le=100, description="Distance from home to work.")
    Education: int = Field(..., ge=1, le=5, description="1 Below College, 2 College, 3 Bachelor, 4 Master, 5 Doctor.")
    EducationField: Literal[
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ]
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="1 Low, 2 Medium, 3 High, 4 Very High.")
    Gender: Literal["Female", "Male"]
    JobInvolvement: int = Field(..., ge=1, le=4, description="1 Low, 2 Medium, 3 High, 4 Very High.")
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ]
    JobSatisfaction: int = Field(..., ge=1, le=4, description="1 Low, 2 Medium, 3 High, 4 Very High.")
    MaritalStatus: Literal["Divorced", "Married", "Single"]
    MonthlyIncome: int = Field(..., ge=0, le=100000)
    NumCompaniesWorked: int = Field(..., ge=0, le=20)
    OverTime: Literal["No", "Yes"]
    PercentSalaryHike: int = Field(..., ge=0, le=100)
    PerformanceRating: int = Field(..., ge=3, le=4, description="3 Excellent, 4 Outstanding.")
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="1 Low, 2 Medium, 3 High, 4 Very High.")
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=60)
    TrainingTimesLastYear: int = Field(..., ge=0, le=20)
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="1 Bad, 2 Good, 3 Better, 4 Best.")
    YearsAtCompany: int = Field(..., ge=0, le=50)
    YearsInCurrentRole: int = Field(..., ge=0, le=50)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=50)
    YearsWithCurrManager: int = Field(..., ge=0, le=50)


class PredictionResponse(BaseModel):
    attrition: Literal["Yes", "No"]
    probability_attrition: float = Field(..., ge=0, le=1)
    probability_retention: float = Field(..., ge=0, le=1)
    risk_category: Literal["Low", "Medium", "High"]
    risk_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"medium_at": RISK_MEDIUM_AT, "high_at": RISK_HIGH_AT}
    )


class BatchPredictionRequest(BaseModel):
    employees: list[EmployeeFeatures] = Field(..., min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class DriftRequest(BaseModel):
    employees: list[EmployeeFeatures] = Field(..., min_length=1, max_length=5000)
