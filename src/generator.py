
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

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Check the .env file in the project root."
    )

client = genai.Client(api_key=api_key)


MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]


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

    last_error = None

    for model in MODELS:

        for attempt in range(3):

            try:
                print(
                    f"Trying model: {model} "
                    f"(attempt {attempt + 1}/3)"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text:
                    return response.text

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            except errors.ServerError as e:

                last_error = e

                print(
                    f"Temporary Gemini server error: {e}"
                )

                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

            except Exception as e:

                last_error = e

                print(
                    f"Gemini error with {model}: {e}"
                )

                break

    raise RuntimeError(
        "Gemini is temporarily unavailable. "
        "Please try again after a short time."
    ) from last_error