import os
from langchain_community.document_loaders import TextLoader

DATA_PATH = "data/"

def load_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".txt"):
            loader = TextLoader(path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source_file"] = file

            docs.extend(loaded_docs)

    return docs