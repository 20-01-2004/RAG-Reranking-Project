from generator import generate_answer


query = "What is Retrieval-Augmented Generation?"

context = """
Retrieval-Augmented Generation, or RAG, combines information
retrieval with large language models to generate answers using
external knowledge.

A typical RAG system contains document loading, text chunking,
embedding generation, vector search, retrieval, and language
model generation.
"""


answer = generate_answer(
    query,
    context
)


print("\n==============================")
print("GEMINI GENERATED ANSWER")
print("==============================")

print(answer)