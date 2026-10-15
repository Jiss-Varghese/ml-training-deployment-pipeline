Project: ML Training and Deployment Pipeline

Define model training and deployment pipeline requirements

Build a working project that automatically trains a model, tests it, evaluates its quality, packages it, requires approval before release, and deploys it for predictions

Phase 1 

python3 --version
python3 -m pip --version
Create the project directory

cd ~/Documents
mkdir ml-training-deployment-pipeline
cd ml-training-deployment-pipeline

code .

Create a Python virtual environment

python3 -m venv .venv

source .venv/bin/activate
which python
mkdir -p data src tests artifacts docs .github/workflows
touch src/__init__.py
touch .gitignore
touch requirements.txt
touch README.md
touch docs/pipeline-requirements.md

```text
ml-training-deployment-pipeline/
├── .venv/
├── data/
├── src/
│   └── __init__.py
├── tests/
├── artifacts/
├── docs/
│   └── pipeline-requirements.md
├── .github/
│   └── workflows/
├── .gitignore
├── requirements.txt
└── README.md

```

Configure .gitignore
code .gitignore 


.venv/
__pycache__/
.pytest_cache/
.env
mlruns/
artifacts/
.DS_Store


verify:

pwd
python --version
python -m pip --version
ls -la

Phase 2 — Install Python dependencies

Update pip

python -m pip install --upgrade pip
Add the project dependencies

code requirements.txt


scikit-learn
pandas
numpy
mlflow
pytest
joblib
fastapi
uvicorn
httpx


Install the dependencies

python -m pip install -r requirements.txt

Verify the installation:

python -m pip show scikit-learn pandas numpy mlflow pytest joblib fastapi uvicorn httpx

Then check that Python can import the packages:


python -c "import sklearn, pandas, numpy, mlflow, pytest, joblib, fastapi, uvicorn, httpx; print('All dependencies installed successfully')"

should see:

All dependencies installed successfully

Phase 3 — Prepare the dataset
Create the data preparation script
Create a new file named src/prepare_data.py
```text

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def prepare_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame

    # Give the target column a clear name.
    df = df.rename(columns={"target": "species_id"})

    # Check basic data quality.
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if df["species_id"].nunique() != 3:
        raise ValueError("Expected three flower classes.")

    # Save the complete dataset for reference.
    dataset_path = DATA_DIR / "iris.csv"
    df.to_csv(dataset_path, index=False)

    # Split the data while preserving class proportions.
    X = df.drop(columns=["species_id"])
    y = df["species_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["species_id"] = y_train

    test_df = X_test.copy()
    test_df["species_id"] = y_test

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("Dataset preparation completed.")
    print(f"Total records: {len(df)}")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")
    print(f"Saved dataset to: {dataset_path}")


if __name__ == "__main__":
    prepare_data()
```

python src/prepare_data.py

ls -lh data
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').head())"

Phase 4 — Build the model training script
code src/train.py
```text
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def train_model():
    # Create the output directory.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the training and testing datasets.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    target_column = "species_id"

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Build and train the model.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on data it did not train on.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel evaluation")
    print("----------------")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    # Save the trained model.
    model_path = ARTIFACTS_DIR / "iris_model.joblib"
    joblib.dump(model, model_path)

    # Save evaluation metrics for later reporting.
    metrics_path = ARTIFACTS_DIR / "metrics.txt"
    metrics_path.write_text(
        f"model: LogisticRegression\n"
        f"test_accuracy: {accuracy:.4f}\n"
        f"training_records: {len(X_train)}\n"
        f"testing_records: {len(X_test)}\n"
    )

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train_model()

```
python src/train.py
ls -lh artifacts

python -c "import joblib; model=joblib.load('artifacts/iris_model.joblib'); print('Prediction:', model.predict([[5.1, 3.5, 1.4, 0.2]]))"

Phase 5 — Add automated tests
 Create tests/test_training.py

 code tests/test_training.py

 ```text
 from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_testing_data_exist():
    """Check that the prepared datasets exist."""
    assert (ROOT / "data" / "train.csv").exists()
    assert (ROOT / "data" / "test.csv").exists()


def test_dataset_has_expected_columns():
    """Check the training data has features and the target."""
    train_df = pd.read_csv(ROOT / "data" / "train.csv")

    expected_columns = {
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species_id",
    }

    assert expected_columns.issubset(set(train_df.columns))


def test_dataset_has_no_missing_values():
    """Ensure the training dataset contains no missing values."""
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    assert not train_df.isnull().any().any()


def test_saved_model_can_predict():
    """Check that the saved model loads and returns a prediction."""
    model_path = ROOT / "artifacts" / "iris_model.joblib"
    assert model_path.exists()

    model = joblib.load(model_path)

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]
```
python -m pytest -v
Phase 6 — Add MLflow experiment tracking
Update src/train.py

replace and update the script

```text
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

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
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
```

python src/train.py
Open the MLflow dashboard

open a second terminal and run

cd ~/Documents/ml-training-deployment-pipeline
source .venv/bin/activate
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
http://127.0.0.1:5000

Verify the pipeline still passes its tests
Return to your first terminal and run:

python -m pytest -v
Configure your training script

code src/train.py 
add this 

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model-training")

http://127.0.0.1:5000

should see 

iris-model-training
model-training




