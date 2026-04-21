import numpy as np

def retrieve(query, index, metadata, embed_fn, k=5):
    q_vec = embed_fn(query)

    D, I = index.search(np.array([q_vec]), k)

    results = []
    for i in I[0]:
        results.append(metadata[i])

    return results