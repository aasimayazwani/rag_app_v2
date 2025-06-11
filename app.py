import streamlit as st
import os, json
from datetime import datetime
from dotenv import load_dotenv
import logging

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

# Setup logging
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# 1. Constants and Environment
# ─────────────────────────────────────────────
DEFAULT_DATA_DIR = "data_csv"
UPLOAD_DIR       = "uploaded_csv"
FAISS_DIR        = "faiss_index"

for p in (DEFAULT_DATA_DIR, UPLOAD_DIR, FAISS_DIR):
    os.makedirs(p, exist_ok=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────
# 2. LLM + Prompt Setup
# ─────────────────────────────────────────────
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="Llama3-8b-8192")
embedder = OpenAIEmbeddings()

prompt = ChatPromptTemplate.from_template("""
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
""")

# ─────────────────────────────────────────────
# 3. State Initialization
# ─────────────────────────────────────────────
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
# 4. FAISS Builder from CSV
# ─────────────────────────────────────────────
def build_index_from_csv_folder(folder: str):
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".csv"):
            loader = CSVLoader(file_path=os.path.join(folder, file), encoding="utf-8", csv_args={"delimiter": ","})
            docs.extend(loader.load())
    if not docs:
        return None
    return FAISS.from_documents(docs, embedder)

# ─────────────────────────────────────────────
# 5. Load Existing or Create New FAISS
# ─────────────────────────────────────────────
if st.session_state.vectors is None:
    try:
        st.session_state.vectors = FAISS.load_local(FAISS_DIR, embedder)
    except:
        base = build_index_from_csv_folder(DEFAULT_DATA_DIR)
        if base:
            base.save_local(FAISS_DIR)
            st.session_state.vectors = base

# ─────────────────────────────────────────────
# 6. Page Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="CSV Chatbot", layout="wide")
st.title("📊 Chat with your CSV Data")

# ─────────────────────────────────────────────
# 7. Sidebar for CSV Uploads
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂 Indexed CSV Files")
    csv_files = sorted(
        [os.path.join("data_csv", f) for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".csv")]
      + [os.path.join("uploaded_csv", f) for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
    )
    if csv_files:
        for f in csv_files:
            st.markdown(f"- {os.path.basename(f)}")
    else:
        st.info("No CSV files found.")

    with st.expander("➕ Upload more CSVs"):
        uploads = st.file_uploader("Select CSVs", type="csv", accept_multiple_files=True, key="csv_upload")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("Chat cleared.")

    if st.session_state.chat_history:
        st.download_button("⬇️ Export Chat Log", data=json.dumps(st.session_state.chat_history, indent=2), file_name="chat_log.json")

# ─────────────────────────────────────────────
# 8. Handle Uploads
# ─────────────────────────────────────────────
if uploads:
    existing = {f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".csv")}
    existing |= {f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")}

    new_docs, skipped = [], []
    for file in uploads:
        if file.name in existing:
            skipped.append(file.name)
            continue
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        loader = CSVLoader(file_path=save_path, encoding="utf-8", csv_args={"delimiter": ","})
        new_docs.extend(loader.load())
        existing.add(file.name)

    if new_docs:
        new_store = FAISS.from_documents(new_docs, embedder)
        if st.session_state.vectors:
            st.session_state.vectors.merge_from(new_store)
        else:
            st.session_state.vectors = new_store
        st.session_state.vectors.save_local(FAISS_DIR)
        st.success("✅ New CSV data indexed.")

    if skipped:
        st.warning(f"⚠️ Duplicates skipped: {', '.join(skipped)}")

# ─────────────────────────────────────────────
# 9. Chat Interface
# ─────────────────────────────────────────────
if st.session_state.vectors:
    question = st.chat_input("Ask a question about your CSV data")
    if question:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Thinking..."):
            res = chain.invoke({"input": question})
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": res["answer"]
        })

# ─────────────────────────────────────────────
# 10. Display Chat
# ─────────────────────────────────────────────
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {item['timestamp']}**\n\n{item['question']}")
    with st.chat_message("assistant"):
        st.markdown(item['answer'])
