output "namespace" {
  description = "Namespace that holds the application."
  value       = kubernetes_namespace.attrition.metadata[0].name
}

output "api_service" {
  description = "In-cluster DNS name of the API service."
  value       = "http://${kubernetes_service.api.metadata[0].name}.${kubernetes_namespace.attrition.metadata[0].name}.svc.cluster.local:8000"
}

output "streamlit_service" {
  description = "Streamlit service name."
  value       = kubernetes_service.streamlit.metadata[0].name
}

output "streamlit_access" {
  description = "Command that opens the dashboard from the local machine."
  value       = "kubectl port-forward -n ${kubernetes_namespace.attrition.metadata[0].name} svc/${kubernetes_service.streamlit.metadata[0].name} 8501:8501"
}
