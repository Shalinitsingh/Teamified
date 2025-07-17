# PDF RAG Chat with Google Gemini

A simple Retrieval‑Augmented Generation (RAG) pipeline over PDF documents, powered by Google Gemini and LangChain, with a Streamlit UI.

## Features

- **Upload one or more PDFs** in the sidebar
- **Chunk** your documents into 1 000‑char windows with 100‑char overlap
- **Embed** each chunk using Google Gemini’s text‑embedding model
- **Index** them in a persistent Chroma vector store (first run only)
- **Retrieve** the top 3 most relevant chunks for any user query
- **Generate** an answer with Gemini LLM and show source snippets
- **Conversational**: maintains chat history across questions


## Prerequisites

- Python 3.8 or higher  
- A Google Generative AI API key (Gemini)  
- (Optional) GPU for faster embedding, but CPU works too  



