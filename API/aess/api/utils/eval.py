import re
import os
import json
import requests
import mimetypes
import google.generativeai as genai
from .load_creds import load_creds
from io import BytesIO
from pdfminer.high_level import extract_text
from docx import Document

def load_file_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        ext = mimetypes.guess_extension(content_type)
        
        if ext in [".docx", ".doc"]:
            return extract_docx_text(BytesIO(response.content))
        elif ext == ".pdf":
            return extract_pdf_text(BytesIO(response.content))
        elif ext == ".txt":
            return response.text
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error retrieving file: {e}"

def extract_docx_text(file_stream):
    try:
        doc = Document(file_stream)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {e}"

def extract_pdf_text(file_stream):
    try:
        return extract_text(file_stream)
    except Exception as e:
        return f"Error extracting PDF: {e}"

genai.configure(credentials=load_creds())

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/aesmodeltest4-oioi4ubuqer6",
    generation_config=generation_config,
)

def clean_response(response):
    return response.replace("**", "").replace("  ", " ")

def make_prompt(submission_content, description_content, rubric_content):
    return (
        f"DESCRIPTION: {description_content}\n"
        f"RUBRIC: {rubric_content}\n"
        f"CONTENT: {submission_content}\n\n"
        f"PROMPT: Provide an evaluation of the given CONTENT based on the DESCRIPTION and RUBRIC. "
        f"Your response should follow this format exactly:\n\n"
        f"Score: (a single number from 1 to 6)\n"
        f"Feedback: (A clear and concise textual feedback, without any embedded numbers or scores.)"
    )

def extract_score_and_feedback(response_text):
    score = None
    feedback = ""

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+", line)  # Extract the first number
            if match:
                score = int(match.group(0))
        elif line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()  # Extract feedback after "Feedback:"

    if score is None:
        score = 0  # Default if parsing fails

    # 🛠 Remove trailing numbers from the feedback
    feedback = re.sub(r"\d+$", "", feedback).strip()

    return score, feedback

def evaluate_submissions(data):
    # with open(input_json_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    results = []
    
    description_content = "\n".join([load_file_content(url) for entry in data.get("description", []) for url in entry.get("description_urls", [])])
    rubric_content = "\n".join([load_file_content(url) for entry in data.get("rubrics", []) for url in entry.get("rubric_urls", [])])
    
    for submission in data.get("submissions", []):
        submission_id = submission["submission_id"]
        submission_urls = submission["submission_urls"]
        
        submission_texts = [load_file_content(url) for url in submission_urls]
        submission_content = "\n".join(submission_texts)
        
        try:
            prompt = make_prompt(submission_content, description_content, rubric_content)
            response = model.generate_content(prompt)
            response_text = clean_response(response.text)

            score, feedback = extract_score_and_feedback(response_text)

            results.append({
                "submission_id": submission_id,
                "score": score,
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing submission_id {submission_id}: {e}")
            results.append({
                "submission_id": submission_id,
                "score": 0,
                "feedback": "Error processing submission."
            })
    
    print(f"Evaluation complete. Results saved")
    return {"results": results}
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=4)