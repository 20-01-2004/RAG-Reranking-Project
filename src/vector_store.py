import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Create a FAISS index using the given embeddings.
    """

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(index, query_embedding, top_k=2):
    """
    Search the FAISS index and return the most similar chunks.
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    return scores[0], indices[0]