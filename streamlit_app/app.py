"""Employee Attrition Intelligence dashboard.

The dashboard calls FastAPI for predictions, metrics, and drift checks.
It does not load the model file.
"""

from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_FIELDS = [
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

CATEGORICAL_FIELDS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

FIELD_LABELS = {
    "Age": "Age",
    "BusinessTravel": "Business travel",
    "Department": "Department",
    "DistanceFromHome": "Distance from home",
    "Education": "Education level",
    "EducationField": "Education field",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "Gender": "Gender",
    "JobInvolvement": "Job involvement",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "JobSatisfaction": "Job satisfaction",
    "MaritalStatus": "Marital status",
    "MonthlyIncome": "Monthly income",
    "NumCompaniesWorked": "Number of companies worked",
    "OverTime": "Overtime",
    "PercentSalaryHike": "Percent salary hike",
    "PerformanceRating": "Performance rating",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "StockOptionLevel": "Stock option level",
    "TotalWorkingYears": "Total working years",
    "TrainingTimesLastYear": "Training times last year",
    "WorkLifeBalance": "Work-life balance",
    "YearsAtCompany": "Years at company",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
}

CATEGORY_VALUES = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

NUMERIC_LIMITS = {
    "Age": (18, 80),
    "DistanceFromHome": (0, 100),
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (0, 100000),
    "NumCompaniesWorked": (0, 20),
    "PercentSalaryHike": (0, 100),
    "PerformanceRating": (3, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 60),
    "TrainingTimesLastYear": (0, 20),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 50),
    "YearsInCurrentRole": (0, 50),
    "YearsSinceLastPromotion": (0, 50),
    "YearsWithCurrManager": (0, 50),
}

RISK_THRESHOLD_TEXT = (
    "Risk thresholds use the model's attrition probability. "
    "Low is below 0.35. Medium is from 0.35 up to 0.60. High is 0.60 or above."
)

EDUCATIONAL_DISCLAIMER = (
    "Educational use only. This dashboard demonstrates a model trained on the public IBM HR sample dataset. "
    "It is not a live HR system and its scores are not facts about a real employee."
)

FAIRNESS_WARNING = (
    "Fairness and bias warning: do not use this model as the sole basis for hiring, promotion, pay, or termination. "
    "Predictions can reflect historical patterns in the training file, including patterns tied to attributes such as "
    "gender, marital status, and age. A person must review any employment decision."
)

EDUCATION = {
    1: "1 - Below College",
    2: "2 - College",
    3: "3 - Bachelor",
    4: "4 - Master",
    5: "5 - Doctor",
}
SATISFACTION = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
WORK_LIFE = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
PERFORMANCE = {3: "3 - Excellent", 4: "4 - Outstanding"}

RISK_COPY = {
    "Low": "The model assigns a lower attrition score. That is not a guarantee the employee will stay.",
    "Medium": "The model assigns a middle attrition score. A check-in may be useful. This is not a conclusion that the employee will leave.",
    "High": "The model assigns a higher attrition score. HR may want to review workload, growth, and support. This is not a conclusion that the employee will leave.",
}

HR_CONSIDERATIONS = [
    "Workload and how often overtime is required.",
    "Support from the current manager and time in the current role.",
    "Pay relative to the role, level, and experience.",
    "Work-life balance, travel, and job satisfaction.",
]


class APIError(Exception):
    """The dashboard could not complete a FastAPI call."""


def request_json(method: str, api_url: str, path: str, payload: dict | None = None, timeout: float = 60):
    url = api_url.rstrip("/") + path
    try:
        response = httpx.request(method, url, json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        raise APIError(
            f"Cannot reach the API at {url}. Start FastAPI and check the base URL in the sidebar. Details: {exc}"
        ) from exc
    if response.status_code >= 400:
        raise APIError(f"The API returned HTTP {response.status_code}: {response.text}")
    if path == "/metrics":
        return response.text
    return response.json()


def friendly(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def coded_select(label: str, mapping: dict[int, str], default: int) -> int:
    labels = list(mapping.values())
    selected = st.selectbox(label, labels, index=labels.index(mapping[default]))
    reverse = {text: code for code, text in mapping.items()}
    return reverse[selected]


def read_uploaded_csv(upload) -> pd.DataFrame:
    try:
        frame = pd.read_csv(upload)
    except pd.errors.EmptyDataError as exc:
        raise APIError("The CSV file is empty. Include a header row and at least one employee.") from exc
    except pd.errors.ParserError as exc:
        raise APIError(f"The CSV is malformed and could not be parsed. Check the commas and quotes. Details: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise APIError("The file is not valid UTF-8 text. Save the CSV as UTF-8 and upload it again.") from exc
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
        raise APIError("The CSV file has no columns.")
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")
    if frame.empty:
        raise APIError("The CSV file has no employee rows.")
    return frame


def prepare_upload(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        raise APIError("The CSV file has no employee rows.")
    prepared = frame.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    missing = [column for column in MODEL_FIELDS if column not in prepared.columns]
    if missing:
        labels = ", ".join(f"{friendly(column)} ({column})" for column in missing)
        raise APIError("The CSV is missing required columns: " + labels)
    trimmed = prepared[MODEL_FIELDS].dropna(how="all")
    if trimmed.empty:
        raise APIError("The CSV file has no employee rows.")
    problems: list[str] = []
    for index, row in trimmed.iterrows():
        file_row = int(index) + 2 if str(index).isdigit() else index
        for column in CATEGORICAL_FIELDS:
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                problems.append(f"Row {file_row}: {friendly(column)} is empty.")
                continue
            text = str(value).strip()
            if text not in CATEGORY_VALUES[column]:
                allowed = ", ".join(CATEGORY_VALUES[column])
                problems.append(f"Row {file_row}: {friendly(column)} has an invalid value '{text}'. Allowed values: {allowed}.")
        for column, (lower, upper) in NUMERIC_LIMITS.items():
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                problems.append(f"Row {file_row}: {friendly(column)} is empty.")
                continue
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(number):
                problems.append(f"Row {file_row}: {friendly(column)} is not a number ('{value}').")
                continue
            if float(number) < lower or float(number) > upper or float(number) != int(float(number)):
                problems.append(
                    f"Row {file_row}: {friendly(column)} must be a whole number from {lower} to {upper}."
                )
        if len(problems) >= 12:
            break
    if problems:
        extra = ""
        if len(problems) >= 12:
            extra = " Showing the first 12 problems."
        raise APIError("The CSV has invalid employee data." + extra + " " + " ".join(problems[:12]))
    normalized = trimmed[MODEL_FIELDS].copy()
    for column in NUMERIC_LIMITS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(int)
    for column in CATEGORICAL_FIELDS:
        normalized[column] = normalized[column].astype(str).str.strip()
    return json.loads(normalized.to_json(orient="records"))


def score_records(api_url: str, records: list[dict]) -> list[dict]:
    predictions: list[dict] = []
    for start in range(0, len(records), 1000):
        body = request_json(
            "POST",
            api_url,
            "/predict/batch",
            {"employees": records[start : start + 1000]},
        )
        predictions.extend(body["predictions"])
    return predictions


def page_overview(api_url: str) -> None:
    st.subheader("Overview")
    st.caption("These figures describe the historical dataset served by the API. They are not live HR-system counts.")
    overview = request_json("GET", api_url, "/analytics/overview")
    cards = st.columns(4)
    cards[0].metric("Total employees", overview["total_employees"])
    cards[1].metric("Employees who left", overview["employees_left"])
    cards[2].metric("Employees who stayed", overview["employees_stayed"])
    cards[3].metric("Historical attrition", f"{overview['attrition_percentage']:.2f}%")

    distribution = overview["attrition_distribution"]
    pie_frame = pd.DataFrame(
        {
            "Outcome": ["Left" if name == "Yes" else "Stayed" for name in distribution],
            "Employees": list(distribution.values()),
        }
    )
    pie = px.pie(
        pie_frame,
        names="Outcome",
        values="Employees",
        color="Outcome",
        color_discrete_map={"Left": "#b42318", "Stayed": "#0f6e6b"},
        title="Attrition distribution",
    )
    pie.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(pie, use_container_width=True)

    left, right = st.columns(2)
    department = pd.DataFrame(overview["department_attrition"])
    department_chart = px.bar(
        department,
        x="name",
        y="attrition_percentage",
        text="attrition_percentage",
        title="Department-wise attrition",
        labels={"name": "Department", "attrition_percentage": "Attrition %"},
    )
    department_chart.update_traces(marker_color="#0f6e6b")
    left.plotly_chart(department_chart, use_container_width=True)

    roles = pd.DataFrame(overview["job_role_attrition"])
    role_chart = px.bar(
        roles,
        x="attrition_percentage",
        y="name",
        orientation="h",
        title="Job-role-wise attrition",
        labels={"name": "Job role", "attrition_percentage": "Attrition %"},
    )
    role_chart.update_traces(marker_color="#1d4e89")
    role_chart.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10))
    right.plotly_chart(role_chart, use_container_width=True)

    income = overview["monthly_income"]
    years = overview["years_at_company"]
    income_frame = pd.DataFrame(
        {
            "Monthly income": income["left"] + income["stayed"],
            "Outcome": ["Left"] * len(income["left"]) + ["Stayed"] * len(income["stayed"]),
        }
    )
    years_frame = pd.DataFrame(
        {
            "Years at company": years["left"] + years["stayed"],
            "Outcome": ["Left"] * len(years["left"]) + ["Stayed"] * len(years["stayed"]),
        }
    )
    income_cards = st.columns(4)
    income_cards[0].metric("Mean income, left", f"{income['left_mean']:,.0f}")
    income_cards[1].metric("Mean income, stayed", f"{income['stayed_mean']:,.0f}")
    income_cards[2].metric("Median income, left", f"{income['left_median']:,.0f}")
    income_cards[3].metric("Median income, stayed", f"{income['stayed_median']:,.0f}")
    st.plotly_chart(
        px.box(income_frame, x="Outcome", y="Monthly income", color="Outcome", title="Monthly income analysis"),
        use_container_width=True,
    )
    year_cards = st.columns(4)
    year_cards[0].metric("Mean years, left", f"{years['left_mean']:.2f}")
    year_cards[1].metric("Mean years, stayed", f"{years['stayed_mean']:.2f}")
    year_cards[2].metric("Median years, left", f"{years['left_median']:.2f}")
    year_cards[3].metric("Median years, stayed", f"{years['stayed_median']:.2f}")
    st.plotly_chart(
        px.box(years_frame, x="Outcome", y="Years at company", color="Outcome", title="Years at company analysis"),
        use_container_width=True,
    )


def page_predict(api_url: str) -> None:
    st.subheader("Employee prediction")
    st.caption("The score comes from FastAPI. The form labels are plain language. The request still sends the model's column names.")
    st.info(RISK_THRESHOLD_TEXT)
    with st.form("employee_prediction"):
        basic, role, experience = st.columns(3)
        with basic:
            age = st.number_input(FIELD_LABELS["Age"], min_value=18, max_value=80, value=36)
            business_travel = st.selectbox(FIELD_LABELS["BusinessTravel"], CATEGORY_VALUES["BusinessTravel"])
            department = st.selectbox(FIELD_LABELS["Department"], CATEGORY_VALUES["Department"], index=1)
            distance = st.number_input(FIELD_LABELS["DistanceFromHome"], min_value=0, max_value=100, value=7)
            education = coded_select(FIELD_LABELS["Education"], EDUCATION, 3)
            education_field = st.selectbox(FIELD_LABELS["EducationField"], CATEGORY_VALUES["EducationField"], index=1)
            environment = coded_select(FIELD_LABELS["EnvironmentSatisfaction"], SATISFACTION, 3)
            gender = st.selectbox(FIELD_LABELS["Gender"], CATEGORY_VALUES["Gender"])
            involvement = coded_select(FIELD_LABELS["JobInvolvement"], SATISFACTION, 3)
        with role:
            job_level = st.number_input(FIELD_LABELS["JobLevel"], min_value=1, max_value=5, value=2)
            job_role = st.selectbox(FIELD_LABELS["JobRole"], CATEGORY_VALUES["JobRole"], index=6)
            job_satisfaction = coded_select(FIELD_LABELS["JobSatisfaction"], SATISFACTION, 3)
            marital_status = st.selectbox(FIELD_LABELS["MaritalStatus"], CATEGORY_VALUES["MaritalStatus"], index=1)
            monthly_income = st.number_input(FIELD_LABELS["MonthlyIncome"], min_value=0, max_value=100000, value=5000, step=100)
            companies = st.number_input(FIELD_LABELS["NumCompaniesWorked"], min_value=0, max_value=20, value=1)
            overtime = st.selectbox(FIELD_LABELS["OverTime"], CATEGORY_VALUES["OverTime"])
            salary_hike = st.number_input(FIELD_LABELS["PercentSalaryHike"], min_value=0, max_value=100, value=14)
            performance = coded_select(FIELD_LABELS["PerformanceRating"], PERFORMANCE, 3)
        with experience:
            relationship = coded_select(FIELD_LABELS["RelationshipSatisfaction"], SATISFACTION, 3)
            stock = st.number_input(FIELD_LABELS["StockOptionLevel"], min_value=0, max_value=3, value=1)
            working_years = st.number_input(FIELD_LABELS["TotalWorkingYears"], min_value=0, max_value=60, value=10)
            training = st.number_input(FIELD_LABELS["TrainingTimesLastYear"], min_value=0, max_value=20, value=2)
            work_life = coded_select(FIELD_LABELS["WorkLifeBalance"], WORK_LIFE, 3)
            years_company = st.number_input(FIELD_LABELS["YearsAtCompany"], min_value=0, max_value=50, value=5)
            years_role = st.number_input(FIELD_LABELS["YearsInCurrentRole"], min_value=0, max_value=50, value=2)
            years_promotion = st.number_input(FIELD_LABELS["YearsSinceLastPromotion"], min_value=0, max_value=50, value=1)
            years_manager = st.number_input(FIELD_LABELS["YearsWithCurrManager"], min_value=0, max_value=50, value=3)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    values = {
        "Age": int(age),
        "BusinessTravel": business_travel,
        "Department": department,
        "DistanceFromHome": int(distance),
        "Education": int(education),
        "EducationField": education_field,
        "EnvironmentSatisfaction": int(environment),
        "Gender": gender,
        "JobInvolvement": int(involvement),
        "JobLevel": int(job_level),
        "JobRole": job_role,
        "JobSatisfaction": int(job_satisfaction),
        "MaritalStatus": marital_status,
        "MonthlyIncome": int(monthly_income),
        "NumCompaniesWorked": int(companies),
        "OverTime": overtime,
        "PercentSalaryHike": int(salary_hike),
        "PerformanceRating": int(performance),
        "RelationshipSatisfaction": int(relationship),
        "StockOptionLevel": int(stock),
        "TotalWorkingYears": int(working_years),
        "TrainingTimesLastYear": int(training),
        "WorkLifeBalance": int(work_life),
        "YearsAtCompany": int(years_company),
        "YearsInCurrentRole": int(years_role),
        "YearsSinceLastPromotion": int(years_promotion),
        "YearsWithCurrManager": int(years_manager),
    }
    missing = [field for field in MODEL_FIELDS if field not in values]
    if missing:
        st.error("The form is missing model fields: " + ", ".join(missing))
        return

    result = request_json("POST", api_url, "/predict", values)
    color = {"Low": "#0f7b4c", "Medium": "#b86e00", "High": "#b42318"}[result["risk_category"]]
    st.markdown(
        f"<div style='padding:1rem 1.1rem;border-radius:12px;background:{color};color:white;'>"
        f"<div style='font-size:0.9rem;opacity:0.9;'>Predicted attrition</div>"
        f"<div style='font-size:2rem;font-weight:700;'>{result['attrition']}</div>"
        f"<div>Risk category: {result['risk_category']}</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns(2)
    left.metric("Probability of attrition", f"{result['probability_attrition']:.1%}")
    right.metric("Probability of retention", f"{result['probability_retention']:.1%}")
    st.progress(min(1.0, max(0.0, float(result["probability_attrition"]))))
    st.info(RISK_COPY[result["risk_category"]])
    thresholds = result.get("risk_thresholds", {})
    st.caption(
        "Thresholds returned with this prediction: "
        f"medium starts at {thresholds.get('medium_at', 0.35):.2f}, "
        f"high starts at {thresholds.get('high_at', 0.60):.2f}. "
        + RISK_THRESHOLD_TEXT
    )
    st.caption(
        "This probability is a model estimate, not a certainty. The classifier was trained with "
        "balanced class weights, so the score is a relative risk estimate rather than a guaranteed frequency."
    )

    performance = request_json("GET", api_url, "/model/performance")
    top5 = pd.DataFrame(performance["feature_importance_top5"])
    top5["Feature"] = top5["feature"].map(friendly)
    top5 = top5.rename(columns={"importance": "Importance"})[["Feature", "Importance"]]
    st.subheader("Top five model features")
    st.dataframe(top5, use_container_width=True, hide_index=True)
    st.caption(performance["feature_importance_note"])

    st.subheader("HR considerations")
    st.write("These are general follow-up topics. The model does not show that any of them caused this score.")
    for item in HR_CONSIDERATIONS:
        st.write(f"- {item}")


def page_batch(api_url: str) -> None:
    st.subheader("Batch prediction")
    st.caption(
        "Upload a CSV that uses the dataset column names, such as MonthlyIncome and OverTime. "
        "An Attrition column, if present, is ignored and is not used as an input."
    )
    with st.expander("Column names expected in the CSV"):
        st.dataframe(
            pd.DataFrame({"Label": [FIELD_LABELS[field] for field in MODEL_FIELDS], "CSV column": MODEL_FIELDS}),
            use_container_width=True,
            hide_index=True,
        )
    upload = st.file_uploader("Employee CSV", type=["csv"])
    if upload is None:
        return
    frame = read_uploaded_csv(upload)
    st.write(f"Rows in file: {len(frame)}")
    st.dataframe(frame.head(20), use_container_width=True)
    if st.button("Score uploaded employees", type="primary"):
        records = prepare_upload(frame)
        predictions = score_records(api_url, records)
        result = frame.copy()
        result["PredictedAttrition"] = [item["attrition"] for item in predictions]
        result["ProbabilityAttrition"] = [item["probability_attrition"] for item in predictions]
        result["ProbabilityRetention"] = [item["probability_retention"] for item in predictions]
        result["RiskCategory"] = [item["risk_category"] for item in predictions]
        st.session_state["last_batch_records"] = records
        st.session_state["last_batch_result"] = result
    result = st.session_state.get("last_batch_result")
    if result is not None:
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions",
            result.to_csv(index=False).encode("utf-8"),
            file_name="attrition_predictions.csv",
            mime="text/csv",
        )


def _matrix_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        )
    )
    figure.update_layout(title="Confusion matrix", margin=dict(t=40, b=10))
    return figure


def page_performance(api_url: str) -> None:
    st.subheader("Model performance")
    st.caption("Scores below are the saved holdout evaluation. Precision, recall, and F1 are for employees who left (Attrition = Yes).")
    performance = request_json("GET", api_url, "/model/performance")
    metrics = performance["metrics"]
    cards = st.columns(5)
    cards[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cards[1].metric("Precision", f"{metrics['precision']:.3f}")
    cards[2].metric("Recall", f"{metrics['recall']:.3f}")
    cards[3].metric("F1-score", f"{metrics['f1_score']:.3f}")
    cards[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.info(performance["selection_reason"])

    matrix = performance["confusion_matrix"]
    st.plotly_chart(_matrix_figure(matrix["matrix"], matrix["labels"]), use_container_width=True)

    rows = []
    for label, values in performance["classification_report"].items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    comparison = pd.DataFrame(performance["comparison"]["models"])
    chart = px.bar(
        comparison.melt(
            id_vars="name",
            value_vars=["f1_attrition", "recall_attrition", "roc_auc", "accuracy"],
            var_name="Metric",
            value_name="Score",
        ),
        x="name",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model comparison",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(
        comparison[
            [
                "name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    importance = pd.DataFrame(performance["feature_importance_top5"])
    importance["Feature"] = importance["feature"].map(friendly)
    st.subheader("Top five feature importance scores")
    st.plotly_chart(
        px.bar(
            importance,
            x="importance",
            y="Feature",
            orientation="h",
            title=performance["feature_importance_source"],
        ),
        use_container_width=True,
    )
    st.caption(performance["feature_importance_note"])


def _show_drift(report: dict) -> None:
    status = report.get("status", "not_checked")
    if status == "drift_detected":
        st.error(report.get("message", "Data drift detected."))
    elif status == "warning":
        st.warning(report.get("message", "A smaller distribution change was detected."))
    elif status == "no_drift":
        st.success(report.get("message", "No data drift detected."))
    elif status == "insufficient_data":
        st.warning(report.get("message", "Not enough rows to compare distributions."))
    else:
        st.info(report.get("message", "No drift check has been run."))
    st.info(report.get("disclaimer", ""))
    features = report.get("features") or []
    if features:
        table = pd.DataFrame(features)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download drift report",
            json.dumps(report, indent=2).encode("utf-8"),
            file_name="drift_report.json",
            mime="application/json",
        )


def page_drift(api_url: str) -> None:
    st.subheader("Data drift")
    st.write(
        "Data drift means the incoming employee attributes have a different distribution from the training rows. "
        "Model performance degradation means the predictions became less accurate. This page only checks data drift. "
        "Accuracy cannot be calculated from unlabeled uploads."
    )
    latest = request_json("GET", api_url, "/drift/latest")
    if latest.get("status") != "not_checked":
        st.markdown("**Latest check**")
        _show_drift(latest)

    source = st.radio(
        "Records to compare",
        ["Upload a CSV", "Use the last scored batch"],
        horizontal=True,
    )
    records = None
    if source == "Upload a CSV":
        upload = st.file_uploader("Records for drift check", type=["csv"], key="drift_csv")
        if upload is not None:
            records = prepare_upload(read_uploaded_csv(upload))
    else:
        records = st.session_state.get("last_batch_records")
        if not records:
            st.caption("Score a CSV on the Batch prediction page first, or upload a file here.")

    if records and st.button("Run drift check", type="primary"):
        report = request_json("POST", api_url, "/drift/check", {"employees": records})
        st.session_state["drift_report"] = report
    if st.session_state.get("drift_report"):
        st.markdown("**This check**")
        _show_drift(st.session_state["drift_report"])


def main() -> None:
    st.set_page_config(page_title="Employee Attrition Intelligence", page_icon=":bar_chart:", layout="wide")
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.2rem; }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e6ebf2;
                border-radius: 12px;
                padding: 0.55rem 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <h1 style="margin:0; letter-spacing:0.04em; font-size:1.8rem;">EMPLOYEE ATTRITION INTELLIGENCE</h1>
        <p style="margin:0.25rem 0 1rem 0; color:#4b5563;">
        Historical attrition patterns and individual risk scores for HR review.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.warning(EDUCATIONAL_DISCLAIMER)
    st.error(FAIRNESS_WARNING)
    default_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    api_url = st.sidebar.text_input("API base URL", value=default_url)
    section = st.sidebar.radio(
        "Sections",
        ["Overview", "Employee prediction", "Batch prediction", "Model performance", "Data drift"],
    )
    try:
        health = request_json("GET", api_url, "/health", timeout=5)
        if health.get("model_available"):
            st.sidebar.success("API connected. Model loaded.")
        else:
            st.sidebar.error("API is running, but the model is not loaded.")
    except APIError as exc:
        st.sidebar.error("API not connected.")
        st.sidebar.caption(str(exc))

    pages = {
        "Overview": page_overview,
        "Employee prediction": page_predict,
        "Batch prediction": page_batch,
        "Model performance": page_performance,
        "Data drift": page_drift,
    }
    try:
        pages[section](api_url)
    except APIError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
