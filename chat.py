# ─────────────────────────────────────────────
# chat.py – Prompt and LLM Chain
# ─────────────────────────────────────────────
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
""")

def build_chat_chain(vectorstore):
    return create_retrieval_chain(
        vectorstore.as_retriever(),
        create_stuff_documents_chain(llm, prompt)
    )
