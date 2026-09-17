from pdf_loader import load_pdf
from chunker import chunk_text

pdf_path = "data/documents/sample.pdf"

pages = load_pdf(pdf_path)

for page in pages:
    chunks = chunk_text(page["text"], chunk_size=50, overlap=10)

    print("Source:", page["source"])
    print("Page:", page["page"])
    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)