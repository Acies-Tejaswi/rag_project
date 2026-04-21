import streamlit as st
from src.embedding import get_embeddings
from src.vectorstore import load_vectorstore
from src.retrieval import retrieve_docs
from src.rerank import rerank_docs
from src.llm import generate_answer

st.title("RAG Chatbot with Sources")

query = st.text_input("Ask a question:")

if query:
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)

    docs = retrieve_docs(vectorstore, query, k=5)
    docs = rerank_docs(docs)

    answer, sources = generate_answer(query, docs)

    st.write("### Answer")
    st.write(answer)

    st.write("### Sources")
    for s in sources:
        st.write(s)