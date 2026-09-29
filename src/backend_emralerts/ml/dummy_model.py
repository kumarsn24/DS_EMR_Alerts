import numpy as np


class DummyModel:
    """A tiny placeholder model for EMR alert scoring.

    This model sums features and uses a simple rule to generate an alert.
    """
    def predict(self, features):
        arr = np.array(features, dtype=float)
        score = float(np.sum(arr))
        # alert if sum > threshold_multiplier * number_of_features
        return {"score": score, "alert": score > 0.5 * arr.size}
