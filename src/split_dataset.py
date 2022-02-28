from pathlib import Path
import pandas as pd
from constants import (
    CLEANED_DATASET_FILE,
    SPLITTED_DATASET_PATH,
    TRAIN_FEATURES_FILE,
    TRAIN_TARGET_FILE,
    TEST_FEATURES_FILE,
    TEST_TARGET_FILE,
)
from sklearn.model_selection import train_test_split


def split_dataset():
    df = pd.read_parquet(CLEANED_DATASET_FILE)
    df = df[["OCR_text", "Client"]]
    X_train, X_test, y_train, y_test = train_test_split(
        df[["OCR_text"]], df[["Client"]], random_state=0, stratify=df["Client"]
    )
    print(X_train)
    print(X_test)
    X_train.to_parquet(TRAIN_FEATURES_FILE, index=False)
    X_test.to_parquet(TEST_FEATURES_FILE, index=False)
    y_train.to_parquet(TRAIN_TARGET_FILE, index=False)
    y_test.to_parquet(TEST_TARGET_FILE, index=False)


if __name__ == "__main__":
    Path(SPLITTED_DATASET_PATH).mkdir(exist_ok=True, parents=True)
    split_dataset()
