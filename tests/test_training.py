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
