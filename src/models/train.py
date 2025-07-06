from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
import joblib
import json 

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]
    X_test  = test.drop(columns=["Churn"])
    y_test  = test["Churn"]
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression(solver="liblinear", max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(f"F1 score: {f1:.3f}, Recall: {rec:.3f}")

    joblib.dump(model, MODEL_DIR / "baseline.pkl")
    metrics = {"f1_score": f1, "recall": rec}
    with open(MODEL_DIR / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"✅ Metrics saved to {MODEL_DIR / '/metrics.json'}")
    # Save the model with a descriptive name
    print(f"✅ Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_and_evaluate()