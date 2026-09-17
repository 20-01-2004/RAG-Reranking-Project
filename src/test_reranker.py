from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embeddings
from vector_store import create_vector_store, search_vector_store
from reranker import rerank


# 1. Load PDF
pdf_path = "data/documents/sample.pdf"

pages = load_pdf(pdf_path)


# 2. Create chunks
all_chunks = []

for page in pages:
    chunks = chunk_text(
        page["text"],
        chunk_size=50,
        overlap=10
    )

    all_chunks.extend(chunks)


# 3. Create embeddings
document_embeddings = create_embeddings(all_chunks)


# 4. Create FAISS vector store
index = create_vector_store(document_embeddings)


# 5. User query
query = "What is Retrieval-Augmented Generation?"


# 6. Create query embedding
query_embedding = create_embeddings([query])


# 7. Initial FAISS retrieval
scores, indices = search_vector_store(
    index,
    query_embedding,
    top_k=2
)


# 8. Get retrieved chunks
retrieved_chunks = [
    all_chunks[index_number]
    for index_number in indices
]


print("\n==============================")
print("INITIAL FAISS RETRIEVAL")
print("==============================")

for rank, (score, chunk) in enumerate(
    zip(scores, retrieved_chunks),
    start=1
):
    print(f"\nResult {rank}")
    print("FAISS Score:", score)
    print("Chunk:", chunk)


# 9. Rerank retrieved chunks
reranked_results = rerank(
    query,
    retrieved_chunks,
    top_k=2
)


print("\n==============================")
print("AFTER RERANKING")
print("==============================")

for rank, (chunk, score) in enumerate(
    reranked_results,
    start=1
):
    print(f"\nResult {rank}")
    print("Reranker Score:", score)
    print("Chunk:", chunk)