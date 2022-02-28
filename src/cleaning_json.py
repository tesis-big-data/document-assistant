from pathlib import Path
import os
import pandas as pd
import simplejson as json
from constants import (
    CLEANED_DATASET_FILE,
    CLEANED_DATASET_PATH,
    INFERENCE_CURRENT_EXEC_PATH,
    RAW_DOCUMENTS_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
)

MIN_CONF = 50


def get_folders_to_process():
    root = RAW_DOCUMENTS_PATH
    return [
        item
        for item in os.listdir(root)
        if os.path.isdir(os.path.join(root, item))
        and len(os.listdir(os.path.join(root, item))) > 10
    ]


def remove_unconfident_words(doc_conf, doc_text):
    idx_to_remove = []
    for idx, conf in enumerate(doc_conf):
        if int(float(conf)) < MIN_CONF or doc_text[idx] == " ":
            idx_to_remove.append(idx)

    for index in sorted(idx_to_remove, reverse=True):
        del doc_text[index]

    return doc_text


def concat_text(json_doc):
    doc_conf = json_doc["OCR_Data"]["conf"]
    doc_text = json_doc["OCR_Data"]["text"]
    doc_text = remove_unconfident_words(doc_conf, doc_text)
    return " ".join(doc_text)


def clean_documents_training():
    # Open a file
    dirs = os.listdir(RAW_DOCUMENTS_PATH)

    # This would print all the files and directories
    df = pd.DataFrame()
    data = []

    folders_to_process = get_folders_to_process()

    for folder in dirs:
        print(f"Cleaning {folder} documents...")
        if folder in folders_to_process:
            file_dirs = os.listdir(RAW_DOCUMENTS_PATH + "/" + folder)

            for file in file_dirs:
                doc = open(f"{RAW_DOCUMENTS_PATH}/{folder}/{file}")
                doc_text = concat_text(json.load(doc))
                data.append({"Client": folder, "OCR_text": doc_text})

    df = df.append(data, ignore_index=True)
    df["OCR_text"] = df["OCR_text"].replace("\s+", " ", regex=True).str.lower()

    df["Category_Id"] = df["Client"].factorize()[0]

    print(df.tail(20))
    df.to_parquet(CLEANED_DATASET_FILE, index=False)


def clean_documents_inference():
    df = pd.DataFrame()
    data = []
    file_dirs = os.listdir(INFERENCE_CURRENT_EXEC_JSON_PATH)

    print(f"Cleaning inference documents...")
    for file in file_dirs:
        doc = open(f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{file}")
        json_doc = json.load(doc)
        doc_text = concat_text(json_doc)
        data.append(
            {
                "document_id": json_doc["document_id"],
                "OCR_text": doc_text,
            }
        )

    df = df.append(data, ignore_index=True)
    df["OCR_text"] = df["OCR_text"].replace("\s+", " ", regex=True).str.lower()

    print(df.tail(20))
    df.to_parquet(f"{INFERENCE_CURRENT_EXEC_PATH}/dataset.parquet", index=False)


if __name__ == "__main__":
    Path(CLEANED_DATASET_PATH).mkdir(exist_ok=True, parents=True)
    clean_documents_training()
