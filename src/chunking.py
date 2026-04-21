from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks