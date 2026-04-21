from openai import OpenAI
from src.config import XAI_API_KEY

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

def generate_answer(query, docs):
    context = ""
    sources = []

    for d in docs:
        context += d["text"] + "\n\n"
        sources.append(
            f"File: {d['source_file']} | Chunk: {d['chunk_id']}"
        )

    prompt = f"""
    Answer ONLY from context.

    Context:
    {context}

    Question:
    {query}
    """

    res = client.chat.completions.create(
        model="grok-2-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content, sources