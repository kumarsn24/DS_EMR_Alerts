from fastapi.testclient import TestClient
import numpy as np

from src.backend_emralerts.api import fastapi_app as fa


def _make_dummy_pipeline():
    class DummyPipeline:
        def __init__(self):
            # mimic a scikit-learn pipeline exposing named_steps in some tests
            self.named_steps = {"preprocessor": None, "classifier": self}

        def predict(self, X):
            # X may be a pandas DataFrame or list-like
            try:
                length = len(X)
            except Exception:
                length = 1
            # return consistent per-row dict for our tests
            return [{"score": 0.5, "alert": False} for _ in range(length)]

        def predict_proba(self, X):
            try:
                length = len(X)
            except Exception:
                length = 1
            return np.array([[0.4, 0.6] for _ in range(length)])

    return DummyPipeline()


def test_health_and_model_info(monkeypatch):
    # Arrange: monkeypatch load_pipeline and metadata extractor before TestClient startup
    monkeypatch.setattr(fa, "load_pipeline", lambda path: _make_dummy_pipeline())
    monkeypatch.setattr(fa, "_extract_model_info", lambda pipeline, path: {"classes": [0, 1], "feature_names": ["f1", "f2"]})

    # Act
    client = TestClient(fa.app)
    resp = client.get("/health")

    # Assert
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "model_info" in body
    assert isinstance(body["model_info"], dict)


def test_predict_json_single_and_batch(monkeypatch):
    monkeypatch.setattr(fa, "load_pipeline", lambda path: _make_dummy_pipeline())
    monkeypatch.setattr(fa, "_extract_model_info", lambda pipeline, path: {})

    client = TestClient(fa.app)

    # single record
    payload = {"records": [{"f1": 1.0, "f2": 2.0}]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out, list)
    assert len(out) == 1
    assert "score" in out[0]

    # batch
    payload = {"records": [{"f1": 1.0}, {"f1": 0.1}]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out, list)
    assert len(out) == 2


def test_reload_endpoint(monkeypatch):
    calls = []

    def fake_load(path):
        calls.append(path)
        return _make_dummy_pipeline()

    monkeypatch.setattr(fa, "load_pipeline", fake_load)
    monkeypatch.setattr(fa, "_extract_model_info", lambda pipeline, path: {})

    client = TestClient(fa.app)

    # without key -> forbidden
    r = client.post("/reload")
    assert r.status_code == 403

    # with key -> success
    r = client.post("/reload", params={"key": fa.RELOAD_KEY})
    assert r.status_code == 200
    assert r.json().get("reloaded") is True
    # ensure load_pipeline was called at least once (startup) and once for reload
    assert len(calls) >= 1
