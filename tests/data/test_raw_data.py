import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def test_csv_can_be_loaded():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty
    assert "customerID" in df.columns

def test_raw_data():
    df = pd.read_csv(DATA_PATH)
    assert df["customerID"].notnull().all(), "customerID should not contain null values"
    assert df["customerID"].is_unique, "customerID should be unique"
    assert df["tenure"].between(0, 72).all(), "tenure should be between 0 and 72"
    assert set(df["Churn"].unique()) <= {"Yes", "No"}
    assert df["MonthlyCharges"].between(0,200).all(), "MonthlyCharges should not contain null values"
    