# Backend_EMRAlerts

Python-based MLOps starter project for EMR alerting (Backend_EMRAlerts).

Getting started

- Create a virtual environment: python -m venv .venv
- Activate and install: pip install -r requirements.txt
- Run the API locally: uvicorn src.backend_emralerts.api.app:app --reload --port 8000

Project layout

- src/backend_emralerts: python package with API and model code
- src/backend_emralerts: python package with API and ML code (api/, ml/)
- tests: pytest tests
- Dockerfile: containerize the service
- .github/workflows/ci.yml: basic CI

Project layout (restructured)

- src/backend_emralerts/api: FastAPI app
- src/backend_emralerts/ml: model and prediction wrappers
- data/: raw and processed data (placeholder)
- model/: persisted model artifacts (placeholder)
- configs/: configuration files
- tests/: pytest test suite
