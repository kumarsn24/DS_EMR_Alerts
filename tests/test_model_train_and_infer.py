import os
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier

from src.backend_emralerts.ml import model_train_and_infer as mti


def test_read_input_file_csv(tmp_path):
    p = tmp_path / "input.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(p, index=False)

    loaded = mti.read_input_file(str(p))
    assert list(loaded.columns) == ["a", "b"]
    assert loaded.shape == (2, 2)


def test_build_preprocessing_pipeline_transforms():
    X = pd.DataFrame({"num": [1, 2, 3], "cat": ["x", "y", "x"]})
    pre = mti.build_preprocessing_pipeline(X)
    Xt = pre.fit_transform(X)
    # Expect 3 rows and at least 1 column after transform
    assert Xt.shape[0] == 3
    assert Xt.shape[1] >= 1


def test_save_load_pipeline_and_run_inference(tmp_path):
    # Create simple training data
    X_train = pd.DataFrame({"feat1": [0, 1, 0, 1], "feat2": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([0, 1, 0, 1], name=mti.TARGET_COL)

    # Simple passthrough preprocessor
    pre = ColumnTransformer(transformers=[("pass", "passthrough", ["feat1", "feat2"])])
    clf = DummyClassifier(strategy="uniform", random_state=0)
    pipeline = Pipeline([("pre", pre), ("classifier", clf)])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    model_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, model_path)

    # Create inference input (no TARGET column)
    input_df = pd.DataFrame({"feat1": [1, 0], "feat2": [5.0, 6.0]})
    input_path = tmp_path / "infer_input.csv"
    input_df.to_csv(input_path, index=False)

    out_csv = tmp_path / "preds.csv"
    result = mti.run_inference(str(model_path), str(input_path), output_path=str(out_csv), probability=True)

    assert "PREDICTION" in result.columns
    # For binary classifier with predict_proba, expect PROBA column or class proba columns
    assert any(c.startswith("PREDICTION_PROBA") or c.startswith("PROBA_CLASS_") for c in result.columns)
    assert out_csv.exists()


def test_compute_roc_and_auc_creates_plot(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    # When plot_path is provided, it should use it directly
    plot_path = tmp_path / "roc.png"
    res = mti.compute_roc_and_auc(y_true=y_true, y_score=y_score, plot_path=str(plot_path))
    assert "auc" in res
    assert plot_path.exists()

    # When plot_path is None, function should save next to MODEL_OUTPUT_PATH
    # Monkeypatch MODEL_OUTPUT_PATH to a temp model file inside tmp_path
    original_model_out = mti.MODEL_OUTPUT_PATH
    try:
        mti.MODEL_OUTPUT_PATH = str(tmp_path / "model_dir" / "model.pkl")
        # Ensure directories do not exist yet; function should create parent and save
        res2 = mti.compute_roc_and_auc(y_true=y_true, y_score=y_score, plot_path=None)
        expected_plot = tmp_path / "model_dir" / "roc_curve.png"
        assert expected_plot.exists()
    finally:
        mti.MODEL_OUTPUT_PATH = original_model_out
