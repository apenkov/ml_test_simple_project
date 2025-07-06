import json
from pathlib import Path

import pytest

MODEL_DIR = Path("models")

@pytest.fixture(autouse=True)
def ensure_model_run(tmp_path, monkeypatch):
    """
    Перед тестом подменим MODEL_DIR на tmp_path, 
    чтобы не вмешиваться в реальную папку.
    """
    monkeypatch.setattr("models.train.MODEL_DIR", tmp_path)
    # Попытаемся запустить тренировку
    import models.train as train
    train.train_and_evaluate()
    yield

def test_kpi_thresholds(tmp_path):
    # 1) Загрузить metrics.json
    mfile = tmp_path / "metrics.json"
    assert mfile.exists(), "metrics.json не найден"
    metrics = json.loads(mfile.read_text())

    # 2) Проверяем минимальные требования
    assert metrics["f1_score"] >= 0.7, "F1 упал ниже базового порога 0.7"
    assert metrics["recall"]   >= 0.6, "Recall упал ниже базового порога 0.6"
