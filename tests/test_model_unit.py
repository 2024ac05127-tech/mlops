import joblib
import pandas as pd

import numpy as np
import joblib
import pandas as pd
import pytest
import numbers
MODEL_PATH = "heart_disease_model.pkl"


@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)


def test_model_loads(model):
    """
    Verify model can be loaded.
    """
    assert model is not None


def test_model_has_predict(model):
    """
    Verify predict method exists.
    """
    assert hasattr(model, "predict")


def test_prediction_returns_output(model):
    """
    Verify prediction executes successfully.
    """

    sample = pd.DataFrame(
        [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    )

    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1

def test_prediction_type(model):

    sample = pd.DataFrame(
        [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    )

    prediction = model.predict(sample)

    assert isinstance(prediction[0], numbers.Number)
def test_prediction_shape(model):

    sample = pd.DataFrame([
    [63,1,3,145,233,1,0,150,0,2.3,0,0,1],
    [67,1,2,160,286,0,0,108,1,1.5,1,3,2],
    [37,1,2,130,250,0,1,187,0,3.5,0,0,2],
    [41,0,1,130,204,0,0,172,0,1.4,2,0,2],
    [56,1,1,120,236,0,1,178,0,0.8,2,0,2]
])

    # sample = pd.read_csv("data/heart.csv").drop("num", axis=1).head(5)

    preds = model.predict(sample)

    assert len(preds) == 5

def test_prediction_class_range(model):

    sample = pd.DataFrame(
        [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    )

    pred = model.predict(sample)

    assert pred[0] in [0, 1]

def test_predict_proba(model):

    if hasattr(model, "predict_proba"):

        sample = pd.DataFrame(
            [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
        )

        probs = model.predict_proba(sample)
        
        assert probs.min() >= 0
        assert probs.max() <= 1
       