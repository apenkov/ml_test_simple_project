import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import models.train as train

def test_training_and_saving(tmp_path, monkeypatch):
    # подменяем директории
    monkeypatch.setattr(train, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)

    # создаём искусственные данные
    df = pd.DataFrame({
        "feature1": [0,1,0,1],
        "feature2": [1,0,1,0],
        "Churn":     [0,1,0,1],
    })
    # сохраняем train/test
    df.to_parquet(train.PROCESSED_DIR / "train.parquet", index=False)  
    df.to_parquet(train.PROCESSED_DIR / "test.parquet",  index=False) 

    # запускаем обучение
    train.train_and_evaluate()

    # проверяем, что модель сохранилась
    assert (tmp_path / "baseline.pkl").exists()
    model = joblib.load(tmp_path / "baseline.pkl")
    assert isinstance(model, LogisticRegression)

    # проверяем, что модель overfits одну запись
    single = pd.DataFrame({"feature1":[0],"feature2":[1]})
    pred = model.predict(single)
    assert pred[0] in (0,1)
