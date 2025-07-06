import pandas as pd
from pathlib import Path
import shutil

import pytest
from data import prepare

@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """
    Перенаправляем PROCESSED_DIR в tmp_path,
    чтобы не мусорить на диске.
    """
    monkeypatch.setattr(prepare, "PROCESSED_DIR", tmp_path)
    yield

def make_toy_df(n_yes: int = 10, n_no: int = 10) -> pd.DataFrame:
    """
    Balanced toy DataFrame for full pipeline:
    - n_yes rows with Churn="Yes"
    - n_no  rows with Churn="No"
    - TotalCharges all valid floats (no 'bad' or '')
    """
    rows = []
    for i in range(n_yes + n_no):
        rows.append({
            "customerID": f"id_{i}",
            "Churn": "Yes" if i < n_yes else "No",
            "gender": "Male" if i % 2 == 0 else "Female",
            "tenure": i + 1,
            # CHANGED: always valid numeric TotalCharges for split stratify
            "TotalCharges": float(i + 1),  # ← CHANGED
        })
    return pd.DataFrame(rows)

def test_clean_df_drops_and_casts():
    """Clean_df should convert and drop invalid TotalCharges."""
    df = pd.DataFrame({
        "TotalCharges": ["5.5", "bad", "", None, "10.0"],
        "Churn": ["Yes", "No", "Yes", "No", "Yes"]
    })
    cleaned = prepare.clean_df(df)
    # two valid: "5.5" and "10.0"
    assert cleaned.shape[0] == 2
    assert cleaned["TotalCharges"].dtype == float


def test_full_pipeline_on_toy_df():
    """
    Full pipeline: clean → encode → split_save.
    Stratify will succeed on balanced toy_df.
    """
    df = make_toy_df()  # ← uses 10 Yes / 10 No by default
    df_clean = prepare.clean_df(df)
    X, y = prepare.encode_features(df_clean)

    # 1) shape and classes
    assert X.shape[0] == 20 and set(y.unique()) == {0, 1}

    # 2) split_save (stratify works since both classes ≥2)
    prepare.split_save(X, y)

    # 3) parquet files exist in temp dir
    files = list(prepare.PROCESSED_DIR.iterdir())
    assert any(f.name == "train.parquet" for f in files)
    assert any(f.name == "test.parquet" for f in files)