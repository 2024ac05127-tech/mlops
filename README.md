#  Heart Disease Prediction using MLOps

## Overview
This repository demonstrates a complete MLOps workflow for predicting heart disease from clinical data. The project covers data preparation, model development, experiment tracking, API deployment, containerization, orchestration, monitoring, and CI/CD.

## Key Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- MLflow experiment tracking
- FastAPI inference service
- Docker and Kubernetes deployment
- Prometheus metrics and Grafana dashboards
- GitHub Actions automation

## Workflow
```text
Dataset
   ↓
Preprocessing
   ↓
EDA
   ↓
Model Training
   ↓
MLflow Tracking
   ↓
FastAPI Service
   ↓
Docker
   ↓
Kubernetes
   ↓
Monitoring (Prometheus + Grafana)
```

## Tech Stack
| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn |
| API | FastAPI |
| Tracking | MLflow |
| Containers | Docker |
| Orchestration | Kubernetes |
| Monitoring | Prometheus & Grafana |
| CI | GitHub Actions |

## Running the Project

```bash
git clone <repository-url>
cd Heart-Disease-MLOps-Pipeline
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` to test the REST API.

## API Endpoint

**POST** `/predict`

Returns:
- Predicted class
- Confidence score
- Risk probability

## Monitoring
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## CI Pipeline
GitHub Actions automates dependency installation, testing, linting, model training, and artifact generation.


### Fastapi Endpoint

![](screenshots/Fastapi_endpoint.png)

### Swagger Ui

![](screenshots/swagger_ui.png)

### Swagger Results

![](screenshots/swagger_results.png)

### Docker Image

![](screenshots/docker_image.png)

### Container Running

![](screenshots/container_running.png)

### Kubectl Get Deployments

![](screenshots/kubectl_get_deployments.png)
### Kubectl Get Svc

![](screenshots/kubectl_get_svc.png)

### Minikube Service

![](screenshots/minikube_service.png)

### Minikube Start

![](screenshots/minikube_start.png)

### Minikube Status

![](screenshots/minikube_status.png)

### Mlflow Runs

![](screenshots/mlflow_runs.png)

### Prometheus Dashboard

![](screenshots/prometheus_dashboard.png)

### Prometheus Query Table

![](screenshots/prometheus_query_table.png)

### Grafana Dashboard

![](screenshots/grafana_dashboard.png)

### Metrics Log

![](screenshots/metrics_log.png)







## Author
**Krishna Priya R**
