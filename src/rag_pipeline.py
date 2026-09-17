from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embeddings
from vector_store import create_vector_store, search_vector_store
from reranker import rerank
from generator import generate_answer


class RAGPipeline:

    def __init__(self, pdf_path):

        # Load PDF
        self.pages = load_pdf(pdf_path)

        # Create chunks
        self.chunks = []

        for page in self.pages:

            page_chunks = chunk_text(
                page["text"],
                chunk_size=50,
                overlap=10
            )

            for chunk in page_chunks:

                self.chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"]
                })

        # Extract chunk texts
        chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Create embeddings
        embeddings = create_embeddings(chunk_texts)

        # Create FAISS index
        self.index = create_vector_store(embeddings)


    def ask(self, query):

        # Create query embedding
        query_embedding = create_embeddings([query])

        # Initial FAISS retrieval
        scores, indices = search_vector_store(
            self.index,
            query_embedding,
            top_k=min(5, len(self.chunks))
        )

        # Get retrieved chunks
        retrieved_chunks = [
            self.chunks[index]
            for index in indices
            if index >= 0
        ]

        # Extract text for reranking
        retrieved_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        # Rerank
        reranked = rerank(
            query,
            retrieved_texts,
            top_k=min(3, len(retrieved_texts))
        )
        print("\n===== RETRIEVED CHUNKS =====")

        for text, score in reranked:
            print("\nScore:", score)
            print("Chunk:", text)

        # Build context
        context_parts = []

        sources = []

        for text, score in reranked:

            matching_chunk = next(
                chunk
                for chunk in retrieved_chunks
                if chunk["text"] == text
            )

            context_parts.append(text)

            sources.append({
                "source": matching_chunk["source"],
                "page": matching_chunk["page"],
                "score": float(score)
            })

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = generate_answer(
            query,
            context
        )

        return {
            "answer": answer,
            "sources": sources
        }
if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    rag = RAGPipeline(pdf_path)

    question = input("\nEnter your question: ")

    result = rag.ask(question)

    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")

    for source in result["sources"]:
        print(
            f"📄 {source['source']} | "
            f"Page {source['page']} | "
            f"Reranker Score: {source['score']:.4f}"
        )