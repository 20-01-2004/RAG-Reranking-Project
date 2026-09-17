from rag_pipeline import RAGPipeline


pdf_path = "data/documents/sample.pdf"


# Create RAG pipeline
rag = RAGPipeline(pdf_path)


# Ask question
query = "What is Retrieval-Augmented Generation?"

result = rag.ask(query)


print("\n==============================")
print("FINAL RAG ANSWER")
print("==============================")

print(result["answer"])


print("\n==============================")
print("SOURCES")
print("==============================")

for source in result["sources"]:

    print(
        f"Source: {source['source']} | "
        f"Page: {source['page']} | "
        f"Reranker Score: {source['score']:.4f}"
    )