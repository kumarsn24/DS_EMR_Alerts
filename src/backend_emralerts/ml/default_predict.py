from src.backend_emralerts.ml.model import DummyModel


class Predictor:
    def __init__(self):
        self.model = DummyModel()

    def predict(self, features):
        # Accept either a single feature list or a list of feature lists
        if not features:
            return []
        if isinstance(features[0], list):
            return [self.model.predict(f) for f in features]
        return self.model.predict(features)
