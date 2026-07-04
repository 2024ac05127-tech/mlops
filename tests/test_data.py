
import pandas as pd

def test_no_null_values():
    df=pd.read_csv("data/heart.csv")
    assert df.isnull().sum().sum() == 0
