
import pandas as pd

def test_no_null_values():
<<<<<<< HEAD
    df=pd.read_csv("heart_disease.csv")
=======
    df=pd.read_csv("data/heart.csv")
>>>>>>> 7d0c7d2141572000d5678a4f3219ad929eab2af9
    assert df.isnull().sum().sum() == 0
