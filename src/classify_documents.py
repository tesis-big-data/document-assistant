import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from constants import (
    INFERENCE_CURRENT_EXEC_CLEANED_DATASET,
    INFERENCE_CURRENT_EXEC_PATH,
    MODEL_FILE,
)


def classify_documents():
    features = np.load(Path(f"{INFERENCE_CURRENT_EXEC_PATH}/encoded_features.npy"))
    documents_df = pd.read_parquet(INFERENCE_CURRENT_EXEC_CLEANED_DATASET)

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(features)

    documents_df["cliente"] = predictions

    documents_df.to_parquet(
        f"{INFERENCE_CURRENT_EXEC_PATH}/classified.parquet", index=False
    )


if __name__ == "__main__":
    classify_documents()
