import streamlit as st
import tempfile
import os
import sys


# Allow Python to find files inside src/
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="RAG with Reranking",
    page_icon="🤖",
    layout="wide"
)


# Title
st.title("🤖 RAG-Based GenAI Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "The system uses FAISS vector retrieval, "
    "Cross-Encoder reranking, and Gemini."
)


# PDF upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# Process PDF
if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                pdf_path = temp_file.name

            rag = RAGPipeline(pdf_path)

            st.session_state.rag = rag
            st.session_state.pdf_name = uploaded_file.name

        st.success("PDF processed successfully!")


# Question section
if "rag" in st.session_state:

    st.subheader("Ask a Question")

    query = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask Question"):

        if query.strip():

            try:

                with st.spinner(
                    "Retrieving, reranking and generating answer..."
                ):

                    result = st.session_state.rag.ask(query)


                # Retrieval comparison
                st.subheader("🔎 Retrieval Comparison")

                col1, col2 = st.columns(2)


                # Without reranking
                with col1:

                    st.markdown(
                        "### ❌ Without Reranking"
                    )

                    st.caption(
                        "FAISS Vector Similarity Retrieval"
                    )

                    for i, chunk in enumerate(
                        result["without_reranking"],
                        start=1
                    ):

                        st.markdown(
                            f"**Chunk {i}**"
                        )

                        st.write(
                            chunk["text"]
                        )

                        st.caption(
                            f"Vector Similarity Score: "
                            f"{chunk['vector_score']:.4f}"
                        )

                        st.divider()


                # With reranking
                with col2:

                    st.markdown(
                        "### ✅ With Reranking"
                    )

                    st.caption(
                        "FAISS Retrieval → Cross-Encoder Reranking"
                    )

                    for i, chunk in enumerate(
                        result["with_reranking"],
                        start=1
                    ):

                        st.markdown(
                            f"**Rank {i}**"
                        )

                        st.write(
                            chunk["text"]
                        )

                        st.caption(
                            f"Reranker Score: "
                            f"{chunk['reranker_score']:.4f}"
                        )

                        st.divider()


                # Final answer
                st.subheader("💡 Final Answer")

                st.write(
                    result["answer"]
                )

                st.info(
                    "The final answer is generated once "
                    "using the reranked context."
                )


            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(e)

        else:

            st.warning(
                "Please enter a question."
            )