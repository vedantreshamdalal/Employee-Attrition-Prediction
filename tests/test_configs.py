"""Static checks for the dashboard fields, monitoring queries, and delivery files."""

import ast
import json
from pathlib import Path

from ml.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_fields_match_the_model_contract():
    module = ast.parse((ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8"))
    found = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_FIELDS":
                    found = ast.literal_eval(node.value)
    assert found == FEATURE_COLUMNS


def test_grafana_queries_match_exported_metric_names():
    dashboard = json.loads((ROOT / "monitoring" / "grafana" / "dashboards" / "attrition.json").read_text(encoding="utf-8"))
    expressions = []
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            expressions.append(target["expr"])
    joined = "\n".join(expressions)
    for name in (
        "prediction_requests_total",
        "attrition_predictions_total",
        "retention_predictions_total",
        "http_request_duration_seconds",
        "failed_requests_total",
        "model_available",
        "process_cpu_seconds_total",
        "process_resident_memory_bytes",
    ):
        assert name in joined
    assert "container_cpu_usage_seconds_total" in joined
    assert "container_memory_usage_bytes" in joined
    titles = [panel["title"] for panel in dashboard["panels"]]
    assert "API process CPU" in titles
    assert "API process resident memory" in titles
    assert "Container CPU (cAdvisor)" in titles
    assert "Container memory (cAdvisor)" in titles
    assert "CPU utilization" not in titles
    assert "Memory utilization" not in titles
    assert len(dashboard["panels"]) == 11


def test_prometheus_scrapes_the_api_metrics_path():
    compose = (ROOT / "monitoring" / "prometheus.yml").read_text(encoding="utf-8")
    cluster = (ROOT / "monitoring" / "prometheus-k8s.yml").read_text(encoding="utf-8")
    for text in (compose, cluster):
        assert "job_name: attrition-api" in text
        assert "metrics_path: /metrics" in text
    assert "api:8000" in compose
    assert "job_name: cadvisor" in compose
    assert "cadvisor:8080" in compose
    assert "job_name: cadvisor" not in cluster
    assert "attrition-api.employee-attrition.svc.cluster.local:8000" in cluster
    datasource = (ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "datasource.yml").read_text(encoding="utf-8")
    assert "uid: prometheus" in datasource
    assert "http://prometheus:9090" in datasource


def test_docker_and_workflow_files_exist():
    api_docker = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    ui_docker = (ROOT / "streamlit_app" / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in api_docker
    assert "EXPOSE 8501" in ui_docker
    for service in ("api:", "streamlit:", "prometheus:", "grafana:"):
        assert service in compose
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cd = (ROOT / ".github" / "workflows" / "cd.yml").read_text(encoding="utf-8")
    assert "pytest" in ci
    assert "docker build -f api/Dockerfile" in ci
    assert "docker build -f streamlit_app/Dockerfile" in ci
    assert "DOCKERHUB_USERNAME" in cd
    assert "DOCKERHUB_TOKEN" in cd
    assert "GitHub-hosted runners cannot reach a local Minikube cluster." in cd
    assert 'tags: ["v*"]' in cd
    assert "Validate semantic version tag" in cd
    assert "vMAJOR.MINOR.PATCH" in cd
    assert 'tags: ["v[0-9]+.[0-9]+.[0-9]+"]' not in cd


def test_kubernetes_and_terraform_define_the_application():
    required = [
        "kubernetes/namespace.yaml",
        "kubernetes/api-deployment.yaml",
        "kubernetes/api-service.yaml",
        "kubernetes/streamlit-deployment.yaml",
        "kubernetes/streamlit-service.yaml",
        "kubernetes/hpa.yaml",
        "terraform/main.tf",
        "terraform/variables.tf",
        "terraform/outputs.tf",
        "terraform/versions.tf",
    ]
    for relative in required:
        assert (ROOT / relative).is_file()
    api_deploy = (ROOT / "kubernetes" / "api-deployment.yaml").read_text(encoding="utf-8")
    ui_deploy = (ROOT / "kubernetes" / "streamlit-deployment.yaml").read_text(encoding="utf-8")
    assert "livenessProbe" in api_deploy and "readinessProbe" in api_deploy
    assert "livenessProbe" in ui_deploy and "readinessProbe" in ui_deploy
    assert "http://attrition-api:8000" in ui_deploy
    terraform = (ROOT / "terraform" / "main.tf").read_text(encoding="utf-8")
    assert 'resource "kubernetes_namespace"' in terraform
    assert 'resource "kubernetes_deployment" "api"' in terraform
    assert 'resource "kubernetes_service" "api"' in terraform
    assert 'resource "kubernetes_deployment" "streamlit"' in terraform
    assert 'resource "kubernetes_service" "streamlit"' in terraform
    assert "hashicorp/kubernetes" in (ROOT / "terraform" / "versions.tf").read_text(encoding="utf-8")


def test_submission_archives_exclude_local_environments():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    exported = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in (".venv/", ".pytest_cache/", ".terraform/", "ml/drift_report.json"):
        assert pattern in ignored
        assert pattern in exported
    assert "model.pkl" not in ignored
    assert ".terraform.lock.hcl" not in ignored
