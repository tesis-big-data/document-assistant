import joblib
import numpy as np
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from constants import (
    ENCODED_DATASET_PATH,
    INFERENCE_CURRENT_EXEC_CLEANED_DATASET,
    INFERENCE_CURRENT_EXEC_PATH,
    TEST_ENCODED_TARGET_FILE,
    TRAIN_ENCODED_FEATURES_FILE,
    TEST_ENCODED_FEATURES_FILE,
    TRAIN_ENCODED_TARGET_FILE,
    TRAIN_TARGET_FILE,
    TEST_TARGET_FILE,
    TRAIN_FEATURES_FILE,
    TEST_FEATURES_FILE,
)


def encode_training():
    X_train = pd.read_parquet(TRAIN_FEATURES_FILE)
    X_test = pd.read_parquet(TEST_FEATURES_FILE)
    y_train = pd.read_parquet(TRAIN_TARGET_FILE)
    y_test = pd.read_parquet(TEST_TARGET_FILE)

    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_enc = tfidf.fit_transform(X_train["OCR_text"]).toarray()

    # Save fitted tfid encoder
    joblib.dump(tfidf, f"{ENCODED_DATASET_PATH}/tfidf.pkl")

    X_test_enc = tfidf.transform(X_test["OCR_text"]).toarray()

    # save features
    np.save(Path(TRAIN_ENCODED_FEATURES_FILE), X_train_enc)
    np.save(Path(TEST_ENCODED_FEATURES_FILE), X_test_enc)

    # save targets
    y_train.to_parquet(TRAIN_ENCODED_TARGET_FILE, index=False)
    y_test.to_parquet(TEST_ENCODED_TARGET_FILE, index=False)


def encode_inference():
    dataset = pd.read_parquet(INFERENCE_CURRENT_EXEC_CLEANED_DATASET)
    trained_tfid = joblib.load(f"{ENCODED_DATASET_PATH}/tfidf.pkl")

    features_matrix = trained_tfid.transform(dataset["OCR_text"]).toarray()

    # save features
    np.save(
        Path(f"{INFERENCE_CURRENT_EXEC_PATH}/encoded_features.npy"), features_matrix
    )


if __name__ == "__main__":
    Path(ENCODED_DATASET_PATH).mkdir(exist_ok=True, parents=True)
    encode_training()
