from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pathlib import Path

output_path = Path("data/documents/sample.pdf")

pdf = canvas.Canvas(str(output_path), pagesize=A4)

pdf.setFont("Helvetica", 16)
pdf.drawString(50, 800, "Introduction to Retrieval-Augmented Generation")

pdf.setFont("Helvetica", 11)

text = [
    "Retrieval-Augmented Generation, or RAG, combines information retrieval",
    "with large language models to generate answers using external knowledge.",
    "",
    "A typical RAG system contains document loading, text chunking, embedding",
    "generation, vector search, retrieval, and language model generation.",
    "",
    "Embeddings represent text as numerical vectors. Similar documents and",
    "queries have similar vector representations.",
    "",
    "Reranking improves retrieval by scoring the relevance of retrieved",
    "documents against the user's query."
]

y = 760

for line in text:
    pdf.drawString(50, y, line)
    y -= 25

pdf.save()

print(f"Sample PDF created: {output_path}")