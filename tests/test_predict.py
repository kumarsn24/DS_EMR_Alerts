from src.backend_emralerts.ml.predict import Predictor


def test_predictor_single_input():
    p = Predictor()
    out = p.predict([0.2, 0.3])
    # DummyModel returns a dict for a single input
    assert isinstance(out, dict)
    assert "score" in out and "alert" in out


def test_predictor_batch_input():
    p = Predictor()
    batch = [[0.1, 0.1], [1.0, 1.0]]
    out = p.predict(batch)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(o, dict) for o in out)


def test_predictor_empty():
    p = Predictor()
    out = p.predict([])
    assert out == []
