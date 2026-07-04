import joblib
import pandas as pd

def test_prediction_shape():

    model = joblib.load("heart_disease_model.pkl")

    sample = pd.read_csv("heart.csv").drop("num", axis=1).head(5)

    preds = model.predict(sample)

    assert len(preds) == 5
