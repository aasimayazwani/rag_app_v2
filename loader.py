# ─────────────────────────────────────────────
# loader.py – CSV Handling Module
# ─────────────────────────────────────────────
import os
import logging
from langchain_community.document_loaders import CSVLoader

DEFAULT_DATA_DIR = "data_csv"
UPLOAD_DIR = "uploaded_csv"

for p in (DEFAULT_DATA_DIR, UPLOAD_DIR):
    os.makedirs(p, exist_ok=True)

def load_csvs_from_folder(folder: str):
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".csv"):
            try:
                loader = CSVLoader(file_path=os.path.join(folder, file), encoding="utf-8", csv_args={"delimiter": ","})
                docs.extend(loader.load())
            except Exception as e:
                logging.warning(f"Failed to load {file}: {e}")
    return docs

def handle_uploads(uploaded_files):
    new_docs, skipped = [], []
    existing = set(os.listdir(DEFAULT_DATA_DIR)) | set(os.listdir(UPLOAD_DIR))

    for file in uploaded_files:
        if file.name in existing:
            skipped.append(file.name)
            continue
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        try:
            loader = CSVLoader(file_path=save_path, encoding="utf-8", csv_args={"delimiter": ","})
            new_docs.extend(loader.load())
        except Exception as e:
            logging.error(f"Failed to process {file.name}: {e}")

    return new_docs, skipped