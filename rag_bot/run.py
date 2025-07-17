# run.py

import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor

# 1) Load & configure your Gemini key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("🔑 Please set GEMINI_API_KEY in your .env")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# 2) PDF → raw text
def get_pdf_text(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text

# 3) 1 000‑char chunks with 100‑char overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    pieces = splitter.split_text(text)
    return [Document(page_content=pc) for pc in pieces]

# 4) Parallel Gemini embeddings
from langchain.embeddings.base import Embeddings

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        with ThreadPoolExecutor(max_workers=8) as exe:
            return list(exe.map(
                lambda t: genai.embed_content(model="models/text-embedding-004", content=t)["embedding"],
                texts
            ))

    def embed_query(self, text):
        return genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]

# 5) Persisted Chroma vectorstore
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

@st.cache_resource
def build_vectorstore(_docs):
    persist_dir = "./chroma_db"
    settings = Settings(is_persistent=True, persist_directory=persist_dir, anonymized_telemetry=False)
    embeddings = GeminiEmbeddings()

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        # reload existing index
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=settings
        )

    # otherwise build & persist
    return Chroma.from_documents(
        _docs,
        embeddings,
        client_settings=settings
    )

# 6) Conversational RAG chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

@st.cache_resource
def build_chain(_vs):
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    retriever = _vs.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

# 7) Streamlit interface
st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("📄 PDF RAG Chat with Gemini")

pdfs = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
if not pdfs:
    st.info("📂 Upload one or more PDFs in the sidebar to get started.")
    st.stop()

# When PDF(s) uploaded, build index + chain
raw = get_pdf_text(pdfs)
docs = get_text_chunks(raw)
vs = build_vectorstore(docs)
chain = build_chain(vs)

# Ask & display in the requested format
q = st.text_input("Ask a question:")
if q and st.button("Send"):
    res = chain({"question": q, "chat_history": []})
    chunks = res["source_documents"]
    answer = res["answer"]

    # 1) User Query
    st.markdown(f"**User Query:** {q}")

    # 2) Retrieved Chunks
    st.markdown("**Retrieved Chunks:**")
    for doc in chunks:
        snippet = doc.page_content.replace("\n", " ").strip()
        # truncate to ~100 chars
        if len(snippet) > 100:
            snippet = snippet[:100].rsplit(" ",1)[0] + "..."
        st.markdown(f'- "{snippet}"')

    # 3) LLM Response
    st.markdown("**LLM Response:**")
    st.markdown(f'"{answer}"')
