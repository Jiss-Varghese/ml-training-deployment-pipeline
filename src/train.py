
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model-training")




from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MLRUNS_DIR = ROOT / "mlruns"


def train_model():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-model-training")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    target = "species_id"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    max_iter = 1000

    with mlflow.start_run():
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        model_path = ARTIFACTS_DIR / "iris_model.joblib"
        joblib.dump(model, model_path)

        metrics_path = ARTIFACTS_DIR / "metrics.txt"
        metrics_path.write_text(
            f"model: LogisticRegression\n"
            f"test_accuracy: {accuracy:.4f}\n"
            f"training_records: {len(X_train)}\n"
            f"testing_records: {len(X_test)}\n"
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_records", len(X_train))
        mlflow.log_param("testing_records", len(X_test))

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions))
        print(f"Model saved to: {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print("MLflow experiment tracking completed.")


if __name__ == "__main__":
    train_model()

