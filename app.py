import streamlit as st
from src.embedding import get_embeddings
from src.vectorstore import load_vectorstore
from src.retrieval import retrieve
from src.rerank import rerank
from src.llm import generate_answer

st.title("RAG Chatbot with Sources")

query = st.text_input("Ask a question:")

if query:
    index, metadata = load_vectorstore()
    docs = retrieve(query, index, metadata, get_embeddings, k=5)

    docs = rerank(query, docs)
    answer, sources = generate_answer(query, docs)

    st.write("### Answer")
    st.write(answer)

    st.write("### Sources")
    for s in sources:
        st.write(s)