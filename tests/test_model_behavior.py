import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def predictor():
    model_path = Path("models") / "baseline.pkl"
    assert model_path.exists(), (
        "Модель не найдена: пожалуйста, сперва выполните "
     "`python src/models/train.py` и закоммитьте файл models/baseline.pkl"
    )
    return joblib.load(model_path)

# 2) Fixture: загружаем тестовый набор признаков
@pytest.fixture(scope="session")
def X_test():
    df = pd.read_parquet(Path("data/processed") / "test.parquet")
    # отбрасываем целевую колонку
    return df.drop(columns=["Churn"])

def test_directional_tenure(predictor, X_test):
    # находим запись, где модель предсказывает 0 («No churn»)
    preds = predictor.predict(X_test)
    idx_no = np.where(preds == 0)[0][0]
    sample = X_test.iloc[[idx_no]].copy()
    # увеличиваем tenure — прогноз должен остаться 0
    sample["tenure"] = sample["tenure"] + 10
    assert predictor.predict(sample)[0] == 0, \
        "При увеличении tenure прогноз NO churn должен сохраниться"
    
def test_proba_sum_to_one(predictor, X_test):
    sample = X_test.iloc[[0]]
    proba = predictor.predict_proba(sample)
    # должно быть две колонки вероятностей
    assert proba.shape == (1, 2)
    # сумма вероятностей ≈ 1
    assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6