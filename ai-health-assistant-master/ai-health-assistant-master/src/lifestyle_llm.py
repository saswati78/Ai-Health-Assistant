from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_lifestyle_advice(disease):

    prompt = f"""
A patient has been predicted with {disease}.

Return ONLY valid JSON (no extra text):

{{
    "is_dangerous": true/false,
    "Diet Plan": "...",
    "Foods to Avoid": "...",
    "Lifestyle": "...",
    "Exercise": "...",
    "Prevention": "..."
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text