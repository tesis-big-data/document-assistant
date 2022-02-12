from pathlib import Path
import os
import pandas as pd
import simplejson as json
from constants import (
    CLEANED_DATASET_FILE,
    CLEANED_DATASET_PATH,
    FOLDERS_TO_PROCESS,
    RAW_DOCUMENTS_PATH,
)

MIN_CONF = 50


def remove_unconfident_words(doc_conf, doc_text):
    idx_to_remove = []
    for idx, conf in enumerate(doc_conf):
        if conf < MIN_CONF or doc_text[idx] == " ":
            idx_to_remove.append(idx)

    for index in sorted(idx_to_remove, reverse=True):
        del doc_text[index]

    return doc_text


def clean_documents():
    # Open a file
    dirs = os.listdir(RAW_DOCUMENTS_PATH)

    # This would print all the files and directories
    df = pd.DataFrame()
    data = []

    for folder in dirs:
        print(f"Cleaning {folder} documents...")
        if folder in FOLDERS_TO_PROCESS:
            file_dirs = os.listdir(RAW_DOCUMENTS_PATH + "/" + folder)
            for file in file_dirs:
                doc = open(f"{RAW_DOCUMENTS_PATH}/{folder}/{file}")
                json_doc = json.load(doc)
                doc_conf = json_doc["ORC_Data"]["conf"]
                doc_text = json_doc["ORC_Data"]["text"]
                doc_text = remove_unconfident_words(doc_conf, doc_text)
                doc_text = " ".join(doc_text)
                data.append({"Client": folder, "OCR_text": doc_text})

    df = df.append(data, ignore_index=True)
    df["OCR_text"] = df["OCR_text"].replace("\s+", " ", regex=True).str.lower()

    df["Category_Id"] = df["Client"].factorize()[0]

    print(df.tail(20))
    df.to_parquet(CLEANED_DATASET_FILE, index=False)


if __name__ == "__main__":
    Path(CLEANED_DATASET_PATH).mkdir(exist_ok=True, parents=True)
    clean_documents()