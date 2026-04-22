import faiss
import numpy as np
import pickle
import os

VECTOR_PATH = "vectorstore/faiss_index.index"
META_PATH = "vectorstore/meta.pkl"

# -------------------------
# CREATE INDEX
# -------------------------
def create_index(docs, embed_fn):

    os.makedirs("vectorstore", exist_ok=True)

    dim = 384
    index = faiss.IndexFlatL2(dim)

    metadata = []

    for doc in docs:
        vec = embed_fn(doc.page_content)

        index.add(np.array([vec]).astype("float32"))

        metadata.append({
            "text": doc.page_content,
            "source_file": doc.metadata["source_file"],
            "chunk_id": doc.metadata["chunk_id"]
        })

    faiss.write_index(index, VECTOR_PATH)
    pickle.dump(metadata, open(META_PATH, "wb"))


# -------------------------
# LOAD INDEX
# -------------------------
def load_vectorstore():
    index = faiss.read_index(VECTOR_PATH)
    metadata = pickle.load(open(META_PATH, "rb"))
    return index, metadata