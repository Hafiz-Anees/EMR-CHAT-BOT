from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os
import certifi
from dotenv import load_dotenv
from refine_query import refine_query
from llm import ask_rag

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Please set GROQ_API_KEY in .env file")
    st.stop()

# streamlit code

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 EMR CHAT BOT")
st.write("Ask about emr chain")



@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("emrchains.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 🔥 Faster + better retrieval model (asymmetric-ready)
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = FAISS.from_documents(
        docs,
        embedding,
        normalize_L2=True  # cosine similarity
    )

    return vector_store


vector_store = load_vectorstore()

# 🔥 Use MMR for better diversity
retriever = vector_store.as_retriever(
    # search_type="mmr",
    search_kwargs={"k": 3}
)


# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask something from the PDF...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get answer
    with st.spinner("Thinking..."):
        answer = ask_rag(user_input)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])