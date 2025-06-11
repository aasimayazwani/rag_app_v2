# ─────────────────────────────────────────────
# index.py – FAISS Index Management
# ─────────────────────────────────────────────
import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from loader import load_csvs_from_folder

FAISS_DIR = "faiss_index"
load_dotenv()

def load_or_create_faiss():
    embedder = OpenAIEmbeddings()
    os.makedirs(FAISS_DIR, exist_ok=True)
    try:
        return FAISS.load_local(FAISS_DIR, embedder)
    except Exception as e:
        logging.warning("No FAISS index found, building new one.")
        docs = load_csvs_from_folder("data_csv")
        if not docs:
            return None
        index = FAISS.from_documents(docs, embedder)
        index.save_local(FAISS_DIR)
        return index

def update_faiss_index(new_docs):
    embedder = OpenAIEmbeddings()
    new_index = FAISS.from_documents(new_docs, embedder)
    existing = load_or_create_faiss()
    if existing:
        existing.merge_from(new_index)
        existing.save_local(FAISS_DIR)
    else:
        new_index.save_local(FAISS_DIR)
