import pandas as pd

DATA_PATH = "data/heart.csv"


def test_dataset_not_empty():
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0


def test_dataset_has_rows_and_columns():
    df = pd.read_csv(DATA_PATH)

    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)

    assert "num" in df.columns


def test_no_duplicate_rows():
    df = pd.read_csv(DATA_PATH)

    assert df.duplicated().sum() == 0


def test_missing_values_percentage():

    df = pd.read_csv(DATA_PATH)

    missing_pct = (
        df.isnull().sum().sum()
        / (df.shape[0] * df.shape[1])
    ) * 100

    assert missing_pct < 5
