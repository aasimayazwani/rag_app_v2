# ─────────────────────────────────────────────
# main.py – Entry Point for Streamlit UI
# ─────────────────────────────────────────────
import streamlit as st
from datetime import datetime
import json

from loader import load_csvs_from_folder, handle_uploads
from index import load_or_create_faiss, update_faiss_index
from chat import build_chat_chain

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="CSV Chatbot", layout="wide")
st.title("📊 Chat with your CSV Data")

# ─────────────────────────────────────────────
# Load or Create FAISS
# ─────────────────────────────────────────────
if st.session_state.vectors is None:
    st.session_state.vectors = load_or_create_faiss()

# ─────────────────────────────────────────────
# Sidebar – Uploads & Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂 Indexed CSV Files")
    st.markdown("""
    Files in `/data_csv` and `/uploaded_csv` will be indexed.
    """)
    uploads = st.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("Chat history cleared.")

    if st.session_state.chat_history:
        st.download_button("⬇️ Download Chat", data=json.dumps(st.session_state.chat_history, indent=2), file_name="chat_log.json")

# ─────────────────────────────────────────────
# Handle File Uploads
# ─────────────────────────────────────────────
if uploads:
    new_docs, skipped = handle_uploads(uploads)
    if new_docs:
        update_faiss_index(new_docs)
        st.success("✅ New CSVs indexed.")
    if skipped:
        st.warning(f"⚠️ Skipped (duplicates): {', '.join(skipped)}")

# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────
if st.session_state.vectors:
    question = st.chat_input("Ask a question about your CSV data")
    if question:
        chain = build_chat_chain(st.session_state.vectors)
        with st.spinner("Thinking..."):
            res = chain.invoke({"input": question})
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": res["answer"]
        })

# ─────────────────────────────────────────────
# Display Chat Log
# ─────────────────────────────────────────────
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {item['timestamp']}**\n\n{item['question']}")
    with st.chat_message("assistant"):
        st.markdown(item['answer'])
