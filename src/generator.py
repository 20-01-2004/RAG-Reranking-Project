
import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


def generate_answer(query, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Do not invent information.

Context:
{context}

User Question:
{query}

Answer clearly and concisely.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text

