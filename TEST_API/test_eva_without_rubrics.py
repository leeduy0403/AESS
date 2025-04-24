import re 
import os
import time
import json
import requests
import mimetypes
import google.generativeai as genai
# from load_creds import load_creds
import pandas as pd
from io import BytesIO
from pdfminer.high_level import extract_text
import pdfplumber
from docx import Document
from vertexai.generative_models import GenerativeModel
import vertexai
from dotenv import load_dotenv

load_dotenv()

# Now read them from the environment
api_key = os.getenv("API_KEY")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Check if they're loaded correctly (optional, for debug only)
print(f"API_KEY loaded: {api_key is not None}")
print(f"Credentials path loaded: {credentials_path is not None}")

# Configure APIs
genai.configure(api_key=api_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

vertexai.init(project="ringed-spirit-422415-v6", location="us-central1")

model = GenerativeModel(
    # model_name="projects/147546299476/locations/us-central1/endpoints/3313124303116959744",     #smaller
    # model_name="projects/147546299476/locations/us-central1/endpoints/7284173274550894592",
    model_name="projects/558798320545/locations/us-central1/endpoints/6503291320411357184", 
    generation_config={
        "temperature": 0.3,
        "top_p": 0.95,
    }
)

feedback_model = genai.GenerativeModel("gemini-2.0-flash")

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

def extract_pdf_text(pdf_path):
    output = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            height = page.height
            # Crop to remove top and bottom 50 pts
            cropped = page.within_bbox((0, 50, page.width, height - 50))
            output.append(cropped.extract_text())
    return "\n\n".join(output)
    
def extract_score_ranges_and_components(description_text):
    """Extracts score ranges and components from the DESCRIPTION text."""
    total_score_match = re.search(r"Total\s+Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)

    # If not found, try the more general "Score range:"
    if not total_score_match:
        total_score_match = re.search(r"Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    
    comp_score_match = re.search(r"Components?\s+Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    components_section = re.search(r"Components:\s*((?:.|\n)*?)(?=(\n[A-Z][^\n]*:|\Z))", description_text, re.IGNORECASE)

    if total_score_match:
        min_total, max_total = map(int, total_score_match.groups())
    else:
        min_total, max_total = 0, 0

    if comp_score_match:
        min_comp, max_comp = map(int, comp_score_match.groups())
    else:
        min_comp, max_comp = 0, 0

    components = []
    if components_section:
        lines = components_section.group(1).splitlines()
        for line in lines:
            comp = re.sub(r"^[\s\-•\d\.o]+", "", line).strip()
            if comp:
                components.append(comp)

    print(f"Total Score Range: {min_total}-{max_total}, Component Score Range: {min_comp}-{max_comp}, Components: {components}")
    return (min_total, max_total), (min_comp, max_comp), components

def clean_feedback(feedback):
    """ Cleans feedback by removing inline numbers and fixing abrupt endings. """
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()

def extract_score(response_text, min_total, max_total, min_comp, max_comp, components):
    """Extracts overall score, individual component scores and feedback."""
    score = min_total
    component_scores = [min_comp] * len(components)

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+(\.\d+)?", line)
            if match:
                score = min(max_total, max(min_total, float(match.group(0))))
                # if score.is_integer():
                #     score = int(score)
        for comp in components:
            pattern = re.compile(rf"{re.escape(comp)}\s*:\s*(\d+(\.\d+)?)", re.IGNORECASE)
            match = pattern.search(response_text)
            if match:
                score_val = float(match.group(1))
                if score_val.is_integer():
                    score_val = int(score_val)
                component_scores[components.index(comp)] = min(max_comp, max(min_comp, score_val))

    # Ensure score is average of components
    # avg_score = round(sum(component_scores) / len(component_scores))
    # score = avg_score

    return score, component_scores


def evaluate_submissions(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    description_urls = data.get("descriptions", [])
    description_content = "\n".join([load_file_content(url) for url in description_urls])
    print(f"description_content: {description_content}")
    (min_total, max_total), (min_comp, max_comp), components = extract_score_ranges_and_components(description_content)
    
    for submission in data.get("submissions", []):
        submission_id = submission["submission_id"]
        submission_urls = submission["submission_urls"]
        
        submission_texts = [load_file_content(url) for url in submission_urls]
        submission_content = "\n".join(submission_texts)
        
        try:
            # Scoring prompt
            prompt = (
                f"Evaluate the given CONTENT based on the DESCRIPTION. For each essay, provide the following:\n"
                f"1. An overall score.\n"
                f"2. A score for each component(if DESCRIPTION has).\n"
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"Follow the exact format below in your response:\n\n"
                f"Score: (a number from {min_total}-{max_total})\n"
            )
            for comp in components:
                prompt += f"{comp}: (a number from {min_comp}-{max_comp})\n"
            
            prompt += (
                f"Example:\n"
                f"Score: {min_total}\n"
            )
            for comp in components:
                prompt += f"{comp}: {min_comp}\n"

            # Feedback prompt
            fb_prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"Your task: Provide **only constructive feedback** on the CONTENT, based on the DESCRIPTION.\n"
                f"Do NOT give any score. Be specific and concise. Mention strengths and areas to improve.\n\n"
                f"If DESCRIPTION has components, provide feedback for each component.\n"
                f"Follow the exact format below in your response:\n\n"
                f"Overall Feedback: (Provide holistic feedback about the essay.)\n"
            )

            for comp in components:
                fb_prompt += f"{comp} Feedback: (Provide specific feedback for the '{comp}' aspect.)\n"

            fb_prompt += (
                f"Example:\n"
                f"Overall Feedback: The essay presents a clear argument with appropriate structure.\n"
            )

            for comp in components:
                fb_prompt += f"{comp} Feedback: Needs more examples.\n"

            score_rp = model.generate_content(prompt)
            score_text = score_rp.text.strip()
            feedback_rp = feedback_model.generate_content(fb_prompt)
            feedback_text = feedback_rp.text.strip()
            print(score_text)
            print(feedback_text)

            score, scores = extract_score(
                score_text, min_total, max_total, min_comp, max_comp, components
            )

            feedback = clean_feedback(feedback_text)

            results.append({
                "submission_id": submission_id,
                "ovr": score,
                "scores": scores,   # array of component scores
                "components": components,
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing submission_id {submission_id}: {e}")
            results.append({
                "submission_id": submission_id,
                "ovr": min_total,
                "scores": [min_comp] * len(components),   # array of component scores
                "components": components,
                "feedback": "Error processing submission."
            })
    
    print("--------------------")
    print(prompt)
    print(f"Evaluation complete. Results saved")
    
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=4)
    
    output_data = {"results": results}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Evaluation complete. Results saved to {output_json_path}")

if __name__ == "__main__":
    input_json = "input.json"
    output_json = "output.json"
    evaluate_submissions(input_json, output_json)
