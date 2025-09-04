import os
from dotenv import load_dotenv
import google.generativeai as genai
from vertexai.generative_models import GenerativeModel

load_dotenv()

model = GenerativeModel(
    model_name="model_name",
    generation_config={
        "temperature": 0.3,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    },
)

feedback_model = genai.GenerativeModel("gemini-2.0-flash")