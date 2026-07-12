import joblib
import pandas as pd

MODEL_PATH = "heart_disease_model.pkl"

def test_end_to_end_prediction():

    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame(
        [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    )

    result = model.predict(input_data)

    assert len(result) == 1

import joblib
import pandas as pd


def test_batch_prediction():

    model = joblib.load(MODEL_PATH)

    X = pd.DataFrame([
        [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
        [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
        [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2]
    ])

    predictions = model.predict(X)

    assert len(predictions) == 3