from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embeddings


pdf_path = "data/documents/sample.pdf"

pages = load_pdf(pdf_path)

all_chunks = []

for page in pages:
    chunks = chunk_text(
        page["text"],
        chunk_size=50,
        overlap=10
    )

    all_chunks.extend(chunks)


embeddings = create_embeddings(all_chunks)

print("Number of chunks:", len(all_chunks))
print("Embedding shape:", embeddings.shape)

for i, embedding in enumerate(embeddings):
    print(f"\nChunk {i + 1}")
    print("First 10 values:")
    print(embedding[:10])