# Terraform is the Kubernetes management path for the namespace, API, and Streamlit.
# Do not also kubectl apply kubernetes/namespace.yaml or the application deployments.
# Monitoring manifests under kubernetes/monitoring.yaml are not managed here.

provider "kubernetes" {
  config_path    = pathexpand(var.kubeconfig_path)
  config_context = var.kube_context
}

resource "kubernetes_namespace" "attrition" {
  metadata {
    name = var.namespace
    labels = {
      app = "employee-attrition"
    }
  }
}

resource "kubernetes_deployment" "api" {
  metadata {
    name      = "attrition-api"
    namespace = kubernetes_namespace.attrition.metadata[0].name
    labels = {
      app = "attrition-api"
    }
  }

  spec {
    replicas = var.api_replicas

    selector {
      match_labels = {
        app = "attrition-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "attrition-api"
        }
      }

      spec {
        container {
          name              = "api"
          image             = var.api_image
          image_pull_policy = var.image_pull_policy

          port {
            name           = "http"
            container_port = 8000
          }

          env {
            name  = "MODEL_PATH"
            value = "/app/ml/model.pkl"
          }

          env {
            name  = "DATA_PATH"
            value = "/app/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
          }

          env {
            name  = "REFERENCE_PATH"
            value = "/app/ml/reference_features.csv"
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 10
            period_seconds        = 10
            timeout_seconds       = 3
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 25
            period_seconds        = 20
            timeout_seconds       = 3
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "1Gi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "api" {
  metadata {
    name      = "attrition-api"
    namespace = kubernetes_namespace.attrition.metadata[0].name
    labels = {
      app = "attrition-api"
    }
  }

  spec {
    selector = {
      app = "attrition-api"
    }

    port {
      name        = "http"
      port        = 8000
      target_port = "http"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_deployment" "streamlit" {
  metadata {
    name      = "attrition-streamlit"
    namespace = kubernetes_namespace.attrition.metadata[0].name
    labels = {
      app = "attrition-streamlit"
    }
  }

  spec {
    replicas = var.streamlit_replicas

    selector {
      match_labels = {
        app = "attrition-streamlit"
      }
    }

    template {
      metadata {
        labels = {
          app = "attrition-streamlit"
        }
      }

      spec {
        container {
          name              = "streamlit"
          image             = var.streamlit_image
          image_pull_policy = var.image_pull_policy

          port {
            name           = "http"
            container_port = 8501
          }

          env {
            name  = "API_BASE_URL"
            value = "http://attrition-api:8000"
          }

          readiness_probe {
            http_get {
              path = "/_stcore/health"
              port = "http"
            }
            initial_delay_seconds = 15
            period_seconds        = 10
            timeout_seconds       = 3
          }

          liveness_probe {
            http_get {
              path = "/_stcore/health"
              port = "http"
            }
            initial_delay_seconds = 30
            period_seconds        = 20
            timeout_seconds       = 3
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "768Mi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "streamlit" {
  metadata {
    name      = "attrition-streamlit"
    namespace = kubernetes_namespace.attrition.metadata[0].name
    labels = {
      app = "attrition-streamlit"
    }
  }

  spec {
    selector = {
      app = "attrition-streamlit"
    }

    port {
      name        = "http"
      port        = 8501
      target_port = "http"
    }

    type = "NodePort"
  }
}
