
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "heart_preprocessing")
TRAIN_PATH = os.path.join(DATA_DIR, "heart_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "heart_test.csv")


def load_data(train_path: str, test_path: str):
    """Memuat dataset train dan test dari file CSV."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test  = test.drop("target", axis=1)
    y_test  = test["target"]

    print(f"Train : {X_train.shape}, Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Melatih model Random Forest dengan parameter default
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Mengevaluasi performa model dan mengembalikan dict metrik.
def evaluate_model(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1_score" : f1_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_pred_prob),
    }

    print("\n=== Hasil Evaluasi Model ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("heart-disease-classification")

    print("Memuat dataset...")
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)

    # Aktifkan autolog sebelum training
    mlflow.sklearn.autolog()

    print("\nMemulai MLflow run dengan autolog...")
    with mlflow.start_run(run_name="RandomForest_baseline"):

        print("Melatih model...")
        model = train_model(X_train, y_train)

        print("\nMengevaluasi model...")
        metrics = evaluate_model(model, X_test, y_test)

    print("\nTraining selesai. Buka MLflow UI:")
    print("  mlflow ui --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
