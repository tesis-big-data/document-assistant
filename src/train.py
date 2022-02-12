import numpy as np
from joblib import dump
from sklearn.metrics import classification_report
from pathlib import Path
from sklearn.svm import LinearSVC
import pandas as pd
from constants import (
    FOLDERS_TO_PROCESS,
    TRAIN_ENCODED_FEATURES_FILE,
    TEST_ENCODED_FEATURES_FILE,
    TRAIN_ENCODED_TARGET_FILE,
    TEST_ENCODED_TARGET_FILE,
    MODEL_FILE,
    MODELS_PATH,
)


def train():
    X_train = np.load(Path(TRAIN_ENCODED_FEATURES_FILE))
    X_test = np.load(Path(TEST_ENCODED_FEATURES_FILE))
    y_train = pd.read_parquet(TRAIN_ENCODED_TARGET_FILE)
    y_test = pd.read_parquet(TEST_ENCODED_TARGET_FILE)

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    print(classification_report(y_test, y_test_pred, target_names=FOLDERS_TO_PROCESS))

    dump(model, MODEL_FILE)


if __name__ == "__main__":
    Path(MODELS_PATH).mkdir(exist_ok=True, parents=True)
    train()