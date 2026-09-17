from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank(query, chunks, top_k=3):
    """
    Rerank retrieved chunks based on their relevance to the query.
    """

    pairs = [[query, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]