from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
app = FastAPI()

model = joblib.load("heart_disease_model.pkl")


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.post("/predict")
def predict(data: HeartData):

    features = np.array([
        [
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]
    ])

    prediction = model.predict(features)[0]

    confidence = float(max(model.predict_proba(features)[0]))

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )    