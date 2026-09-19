# RAG-Based GenAI Assistant with Reranking

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based GenAI assistant that answers questions from uploaded PDF documents.

The system uses FAISS for vector similarity search, Sentence Transformers for generating embeddings, a Cross-Encoder for reranking retrieved documents, and Google Gemini for generating the final answer.

The project also compares document retrieval results before and after reranking.

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Cross-Encoder
- Google Gemini API
- PyMuPDF
- ReportLab

## System Architecture

```text
PDF Document
     ↓
PDF Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Database
     ↓
Vector Similarity Retrieval
     ↓
 ┌───────────────────────────┐
 │                           │
 ↓                           ↓
Without Reranking        Cross-Encoder
                           Reranking
 │                           │
 └─────────────┬─────────────┘
               ↓
        Reranked Context
               ↓
          Gemini LLM
               ↓
        Final Answer
## Key Features

- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate vector embeddings
- Store embeddings using FAISS
- Retrieve relevant document chunks
- Compare retrieval without reranking
- Rerank retrieved chunks using a Cross-Encoder
- Generate grounded answers using Google Gemini
- Streamlit-based user interface

## Reranking Comparison

### Without Reranking

The system retrieves relevant chunks using FAISS vector similarity.

### With Reranking

The retrieved chunks are passed through a Cross-Encoder reranker to improve their relevance to the user's query.

The application displays both results so that the effect of reranking can be observed.

## Project Structure

```text
RAG_Reranking_Project/
│
├── app.py
├── create_sample_pdf.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── documents/
│
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── test_*.py
│
└── screenshots/
    ├── s1.png
    ├── scre-2.png
    ├── scrn-3.png
    └── scrn-4.png