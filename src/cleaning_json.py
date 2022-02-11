from pathlib import Path
import os
import pandas as pd
import simplejson as json

MIN_CONF = 50
FOLDERS_TO_PROCESS = [
    "La Comercial SRL",
    "Modadol",
    "Fernando Garcia",
    "Ayala",
    "Antilur",
    "La Banderita",
    "Los Nietitos",
    "Marioni",
    "Masula",
]


def remove_unconfident_words(doc_conf, doc_text):
    idx_to_remove = []
    for idx, conf in enumerate(doc_conf):
        if conf < MIN_CONF or doc_text[idx] == " ":
            idx_to_remove.append(idx)

    for index in sorted(idx_to_remove, reverse=True):
        del doc_text[index]

    return doc_text


def clean_documents():
    path_to_raw_documents = "assets/json_documents"

    # Open a file
    dirs = os.listdir(path_to_raw_documents)

    # This would print all the files and directories
    df = pd.DataFrame()
    data = []

    for folder in dirs:
        print(f"Cleaning {folder} documents...")
        if folder in FOLDERS_TO_PROCESS:
            file_dirs = os.listdir(path_to_raw_documents + "/" + folder)
            for file in file_dirs:
                doc = open(f"{path_to_raw_documents}/{folder}/{file}")
                json_doc = json.load(doc)
                doc_conf = json_doc["ORC_Data"]["conf"]
                doc_text = json_doc["ORC_Data"]["text"]
                doc_text = remove_unconfident_words(doc_conf, doc_text)
                doc_text = " ".join(doc_text)
                data.append({"Client": folder, "OCR_text": doc_text})

    df = df.append(data, ignore_index=True)
    df["OCR_text"] = df["OCR_text"].replace("\s+", " ", regex=True).str.lower()

    df["Category_Id"] = df["Client"].factorize()[0]

    category_id_df = (
        df[["Client", "Category_Id"]].drop_duplicates().sort_values("Category_Id")
    )
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[["Category_Id", "Client"]].values)

    print(df.tail(20))
    df.to_parquet("assets/cleaned_documents/dataset.parquet", index=False)


if __name__ == "__main__":
    Path("assets/cleaned_documents").mkdir(exist_ok=True, parents=True)
    clean_documents()
    print("HOLA BIENVENIDO A LA TESIS")