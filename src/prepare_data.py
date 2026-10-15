
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
