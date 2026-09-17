from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embeddings
from vector_store import create_vector_store, search_vector_store


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


# 3. Create embeddings for document chunks
document_embeddings = create_embeddings(all_chunks)


# 4. Create FAISS vector store
index = create_vector_store(document_embeddings)

print("FAISS vector store created.")
print("Number of vectors:", index.ntotal)


# 5. Create a user query
query = "What is Retrieval-Augmented Generation?"


# 6. Create embedding for the query
query_embedding = create_embeddings([query])


# 7. Search FAISS
scores, indices = search_vector_store(
    index,
    query_embedding,
    top_k=2
)


# 8. Display results
print("\nUser Query:", query)

for rank, (score, index_number) in enumerate(
    zip(scores, indices),
    start=1
):
    print(f"\n--- Result {rank} ---")
    print("Similarity Score:", score)
    print("Chunk:", all_chunks[index_number])