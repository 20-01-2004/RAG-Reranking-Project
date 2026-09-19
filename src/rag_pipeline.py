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

        # ---------------------------------
        # 1. Create query embedding
        # ---------------------------------

        query_embedding = create_embeddings([query])

        # ---------------------------------
        # 2. FAISS Vector Retrieval
        # ---------------------------------

        scores, indices = search_vector_store(
            self.index,
            query_embedding,
            top_k=min(5, len(self.chunks))
        )

        retrieved_chunks = []

        for score, index in zip(scores, indices):

            index = int(index)

            if index >= 0:

                chunk = self.chunks[index].copy()

                chunk["vector_score"] = float(score)

                retrieved_chunks.append(chunk)

        # Top 3 without reranking
        without_reranking = retrieved_chunks[:3]

        # ---------------------------------
        # 3. Cross-Encoder Reranking
        # ---------------------------------

        retrieved_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        reranked = rerank(
            query,
            retrieved_texts,
            top_k=min(3, len(retrieved_texts))
        )

        # ---------------------------------
        # 4. Prepare reranked chunks
        # ---------------------------------

        with_reranking = []

        for text, reranker_score in reranked:

            matching_chunk = next(
                chunk
                for chunk in retrieved_chunks
                if chunk["text"] == text
            )

            chunk = matching_chunk.copy()

            chunk["reranker_score"] = float(reranker_score)

            with_reranking.append(chunk)

        # ---------------------------------
        # 5. Build final context
        # ---------------------------------

        context_parts = [
            chunk["text"]
            for chunk in with_reranking
        ]

        context = "\n\n".join(context_parts)

        # ---------------------------------
        # 6. ONE Gemini call
        # ---------------------------------

        answer = generate_answer(
            query,
            context
        )

        # ---------------------------------
        # 7. Return results
        # ---------------------------------

        return {
            "answer": answer,
            "without_reranking": without_reranking,
            "with_reranking": with_reranking
        }


if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    rag = RAGPipeline(pdf_path)

    question = input("\nEnter your question: ")

    result = rag.ask(question)

    print("\n===== WITHOUT RERANKING =====")

    for chunk in result["without_reranking"]:

        print(
            f"\nVector Score: "
            f"{chunk['vector_score']:.4f}"
        )

        print(chunk["text"])

    print("\n===== WITH RERANKING =====")

    for chunk in result["with_reranking"]:

        print(
            f"\nReranker Score: "
            f"{chunk['reranker_score']:.4f}"
        )

        print(chunk["text"])

    print("\n===== FINAL ANSWER =====")
    print(result["answer"])