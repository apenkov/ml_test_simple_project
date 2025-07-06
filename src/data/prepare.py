from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the raw data from the CSV file."""
    return pd.read_csv(DATA_PATH)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing unnecessary columns and handling missing values."""
    # Drop columns that are not needed for analysis
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df["Churn"].map({"No": 0, "Yes": 1})
    drop_cols = ["Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols)
    X = df = pd.get_dummies(df, drop_first=True)
    return X, y

def split_save(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train.join(y_train).to_parquet(
        PROCESSED_DIR / "train.parquet", index=False
    )
    X_test.join(y_test).to_parquet(
        PROCESSED_DIR / "test.parquet", index=False
    )
    return X, y

def main():
    df = load_data()
    df = clean_df(df)
    X, y = encode_features(df)
    split_save(X, y)
    print("✅  processed data saved to", PROCESSED_DIR)

if __name__ == "__main__":
    main()
    # Ensure the processed directory exists if not PROCESSED_DIR.exists():
