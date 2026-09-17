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


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG GenAI Assistant",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🤖 RAG-Based GenAI Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "The system uses FAISS retrieval, Cross-Encoder reranking, "
    "and Gemini to generate grounded answers."
)


# -----------------------------
# PDF Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# -----------------------------
# Process PDF
# -----------------------------

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing document..."):

            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                pdf_path = temp_file.name


            # Create RAG pipeline
            rag = RAGPipeline(pdf_path)


            # Store pipeline in session
            st.session_state.rag = rag

            st.session_state.pdf_name = uploaded_file.name

            st.success("PDF processed successfully!")


# -----------------------------
# Question Section
# -----------------------------

if "rag" in st.session_state:

    st.subheader("Ask a Question")

    query = st.text_input(
        "Enter your question:"
    )


    if st.button("Ask Question"):

        if query.strip():

            with st.spinner("Searching documents and generating answer..."):

                result = st.session_state.rag.ask(
                    query
                )


            # -----------------------------
            # Display Answer
            # -----------------------------

            st.subheader("Answer")

            st.write(
                result["answer"]
            )


            # -----------------------------
            # Display Sources
            # -----------------------------


            st.subheader("Sources")

            for source in result["sources"]:

                st.write(
                    f"📄 **{source['source']}** | "
                    f"Page **{source['page']}** | "
                    f"Reranker Score: "
                    f"**{source['score']:.4f}**"
                )

        else:

            st.warning(
                "Please enter a question."
            )