from fastapi.testclient import TestClient
from src.backend_emralerts.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    payload = {"features": [0.1, 0.2, 0.3]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
