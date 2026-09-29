from fastapi import FastAPI
from pydantic import BaseModel
from src.backend_emralerts.ml.predict import Predictor


class InputSchema(BaseModel):
    features: list[float]


app = FastAPI(title="Backend_EMRAlerts")

predictor = Predictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: InputSchema):
    preds = predictor.predict(payload.features)
    return {"predictions": preds}
