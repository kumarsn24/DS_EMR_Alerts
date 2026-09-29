from src.backend_emralerts.ml.model import DummyModel


def test_dummy_model_output_structure():
    m = DummyModel()
    out = m.predict([0.1, 0.2, 0.3])
    assert isinstance(out, dict)
    assert "score" in out and "alert" in out
    assert isinstance(out["score"], float)
    assert isinstance(out["alert"], (bool,))


def test_dummy_model_threshold_behavior():
    m = DummyModel()
    # small values should not trigger alert
    out_low = m.predict([0.0, 0.0, 0.0])
    assert out_low["alert"] is False
    # large values should trigger alert
    out_high = m.predict([10.0, 10.0, 10.0])
    assert out_high["alert"] is True
