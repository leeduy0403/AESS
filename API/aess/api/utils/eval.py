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
    
def extract_score_range(description_text):
    """Extracts the valid score range from the DESCRIPTION text."""
    match = re.search(r"Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    if match:
        min_score, max_score = map(int, match.groups())
    else:
        min_score, max_score = 0, 6  # Default range if not found
    print(f"min_score: {min_score}, max_score: {max_score}")
    return min_score, max_score

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

def clean_feedback(feedback):
    """ Cleans feedback by removing inline numbers and fixing abrupt endings. """
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()

def extract_score_and_feedback(response_text, min_score, max_score):
    """Extracts only the overall score & feedback."""
    score, feedback = min_score, ""
    coherence, lexical, grammar = min_score, min_score, min_score

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+", line)
            if match:
                score = min(max_score, max(min_score, int(match.group(0)))) # Ensure valid range
        if line.lower().startswith("coherence and cohesion:"):
            match = re.search(r"\d+", line)
            if match:
                coherence = min(max_score, max(min_score, int(match.group(0))))  # Ensure valid range
        elif line.lower().startswith("lexical resource:"):
            match = re.search(r"\d+", line)
            if match:
                lexical = min(max_score, max(min_score, int(match.group(0))))
        elif line.lower().startswith("grammatical range and accuracy:"):
            match = re.search(r"\d+", line)
            if match:
                grammar = min(max_score, max(min_score, int(match.group(0))))
        elif line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()

    feedback = clean_feedback(feedback)

    # if score != round((coherence + lexical + grammar) / 3):
    #     score = round((coherence + lexical + grammar) / 3)

    return score, [coherence, lexical, grammar], feedback


def evaluate_submissions(data, output_json_path=None):
    # with open(input_json_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    
    results = []
    
    description_content = "\n".join([load_file_content(url) for entry in data.get("description", []) for url in entry.get("description_urls", [])])
    min_score, max_score = extract_score_range(description_content)
    # rubric_content = "\n".join([load_file_content(url) for entry in data.get("rubrics", []) for url in entry.get("rubric_urls", [])])
    
    for submission in data.get("submissions", []):
        submission_id = submission["submission_id"]
        submission_urls = submission["submission_urls"]
        
        submission_texts = [load_file_content(url) for url in submission_urls]
        submission_content = "\n".join(submission_texts)
        
        try:
            prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION. "
                f"Follow the exact format below in your response:\n\n"
                f"Score: (a number from {min_score}-{max_score} and is the average number of component scores)\n"
                f"Coherence and Cohesion: (a number from {min_score}-{max_score})\n"
                f"Lexical Resource: (a number from {min_score}-{max_score})\n"
                f"Grammatical Range and Accuracy: (a number from {min_score}-{max_score})\n"
                f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)"

                f"Example:\n"
                f"Score: 9\n"
                f"Coherence and Cohesion: 8 \n"
                f"Lexical Resource: 9 \n"
                f"Grammatical Range and Accuracy: 10 \n"
                f"Feedback: The essay presents clear arguments and a strong structure. To enhance clarity, focus on smoother transitions and more precise word choices.\n\n"
            )

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            score, scores, feedback = extract_score_and_feedback(response_text, min_score, max_score)

            results.append({
                "submission_id": submission_id,
                "ovr": score,
                "scores": scores,   # array of component scores
                "components": {
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                },
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing submission_id {submission_id}: {e}")
            results.append({
                "submission_id": submission_id,
                "ovr": min_score,
                "scores": [min_score, min_score, min_score],   # array of component scores
                "components": {
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                },
                "feedback": feedback
            })
    
    print("--------------------")
    print(prompt)
    print(f"Evaluation complete. Results saved")
    return {"results": results}
    
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=4)