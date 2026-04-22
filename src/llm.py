from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def generate_answer(query, docs):
    context = ""
    sources = []

    for d in docs:
        context += d["text"] + "\n\n"
        sources.append(
            f"File: {d['source_file']} | Chunk: {d['chunk_id']}"
        )

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = res.choices[0].message.content

    return answer, sources