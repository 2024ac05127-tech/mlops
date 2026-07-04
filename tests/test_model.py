import joblib
import pandas as pd

def test_prediction_shape():

    model = joblib.load("heart_disease_model.pkl")

<<<<<<< HEAD
    sample = pd.read_csv("heart_disease.csv").drop("num", axis=1).head(5)
=======
    sample = pd.read_csv("data/heart.csv").drop("num", axis=1).head(5)
>>>>>>> 7d0c7d2141572000d5678a4f3219ad929eab2af9

    preds = model.predict(sample)

    assert len(preds) == 5
