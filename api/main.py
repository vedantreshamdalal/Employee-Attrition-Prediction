"""Employee Attrition Intelligence API."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.metrics import (
    failed_requests_total,
    http_request_duration_seconds,
    http_requests_total,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftRequest,
    EmployeeFeatures,
    PredictionResponse,
)
from api.service import (
    ModelNotAvailable,
    dataset_overview,
    latest_drift,
    load_artifacts,
    load_error_message,
    model_is_loaded,
    performance_payload,
    predict_records,
    run_drift,
)

logger = logging.getLogger("attrition-api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    load_artifacts()
    if not model_is_loaded():
        logger.error("Model is not available: %s", load_error_message())
    else:
        logger.info("Model pipeline loaded.")
    yield


app = FastAPI(
    title="Employee Attrition Intelligence API",
    version="1.0.0",
    description=(
        "Score employee attrition risk with the saved scikit-learn pipeline. "
        "Precision-oriented monitoring is exposed for Prometheus."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    endpoint = request.url.path
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()
        http_request_duration_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        if status_code >= 400:
            failed_requests_total.labels(endpoint=endpoint).inc()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _unavailable(exc: ModelNotAvailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
def root() -> dict:
    return {
        "name": "Employee Attrition Intelligence API",
        "health": "/health",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health")
def health():
    available = model_is_loaded()
    payload = {
        "status": "healthy" if available else "unhealthy",
        "model_available": available,
        "detail": None if available else load_error_message(),
    }
    if not available:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures) -> dict:
    try:
        return predict_records([employee.model_dump()])[0]
    except ModelNotAvailable as exc:
        return _unavailable(exc)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    try:
        predictions = predict_records([employee.model_dump() for employee in batch.employees])
    except ModelNotAvailable as exc:
        return _unavailable(exc)
    return {"count": len(predictions), "predictions": predictions}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/analytics/overview")
def overview():
    try:
        return dataset_overview()
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/model/performance")
def model_performance():
    try:
        return performance_payload()
    except ModelNotAvailable as exc:
        return _unavailable(exc)


@app.post("/drift/check")
def drift_check(batch: DriftRequest):
    try:
        return run_drift([employee.model_dump() for employee in batch.employees])
    except ModelNotAvailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/drift/latest")
def drift_latest() -> dict:
    return latest_drift()
