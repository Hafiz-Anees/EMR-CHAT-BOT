import streamlit as st
import os
import certifi
from dotenv import load_dotenv

# -----------------------------
# ENV + SSL FIX
# -----------------------------
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Please set GROQ_API_KEY in .env file")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 EMR CHAT BOT")
st.write("Ask about emr chain")

# -----------------------------
# LOAD & CACHE VECTOR STORE
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


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
# GROQ SETUP
# -----------------------------
from groq import Groq

client = Groq(api_key=groq_api_key)

# -----------------------------
# RAG FUNCTION
# -----------------------------
def ask_rag(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
            "role": "system",
            "content": """
            You are an AI assistant representing EMR Chain.

            Your purpose is to provide accurate, professional, and helpful information about the company.

            Behavior Guidelines:

            1. Identity Questions:
            - If the user asks about your identity (e.g., "Who are you?"), respond:
                "I am an AI assistant for EMR Chain, here to help answer questions about the company."

            2. Company & Services Questions:
            - If the user asks about EMR Chain’s services, impact, or offerings, respond confidently as a representative of the company.
            - Use the provided context as your primary source of information.

            3. Context-Based Answers:
            - For all other questions, answer ONLY using the provided context.
            - Do NOT generate or assume information that is not present in the context.

            4. Unknown Information:
            - If the answer cannot be found in the context, respond with:
                "I don't know based on the provided information."

            5. Tone and Style:
            - Be clear, concise, and professional.
            - Avoid unnecessary verbosity.
            - Provide structured and easy-to-understand responses.
                """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {question}
                    """
            }
        ]
    )

    return response.choices[0].message.content



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