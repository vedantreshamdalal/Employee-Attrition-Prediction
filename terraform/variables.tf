variable "kubeconfig_path" {
  description = "Path to the kubeconfig written by Minikube."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "kubectl context. Minikube's default context name is minikube."
  type        = string
  default     = "minikube"
}

variable "namespace" {
  description = "Namespace managed by Terraform for the attrition application."
  type        = string
  default     = "employee-attrition"
}

variable "api_image" {
  description = "API image. Use the local tag after minikube image load, or a Docker Hub tag."
  type        = string
  default     = "employee-attrition-api:local"
}

variable "streamlit_image" {
  description = "Streamlit image. Use the local tag after minikube image load, or a Docker Hub tag."
  type        = string
  default     = "employee-attrition-streamlit:local"
}

variable "image_pull_policy" {
  description = "IfNotPresent for images loaded into Minikube. Always for a remote registry."
  type        = string
  default     = "IfNotPresent"
}

variable "api_replicas" {
  description = "API pod count."
  type        = number
  default     = 1
}

variable "streamlit_replicas" {
  description = "Streamlit pod count."
  type        = number
  default     = 1
}
