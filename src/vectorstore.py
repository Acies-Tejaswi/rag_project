import faiss
import numpy as np
import pickle

VECTOR_PATH = "vectorstore/faiss_index"

def create_index(docs, embed_fn):
    dim = 384  # MiniLM dimension
    index = faiss.IndexFlatL2(dim)

    metadata = []

    for doc in docs:
        vec = embed_fn(doc.page_content)
        index.add(np.array([vec]))

        metadata.append({
            "text": doc.page_content,
            "source_file": doc.metadata["source_file"],
            "chunk_id": doc.metadata["chunk_id"]
        })

    faiss.write_index(index, VECTOR_PATH)
    pickle.dump(metadata, open("vectorstore/meta.pkl", "wb"))

def load_index():
    index = faiss.read_index(VECTOR_PATH)
    metadata = pickle.load(open("vectorstore/meta.pkl", "rb"))
    return index, metadata