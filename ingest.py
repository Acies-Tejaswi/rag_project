from src.loader import load_documents
from src.chunking import chunk_documents
from src.embedding import get_embedding
from src.vectorstore import create_index

def main():
    docs = load_documents()
    chunks = chunk_documents(docs)

    create_index(chunks, get_embedding)

if __name__ == "__main__":
    main()
