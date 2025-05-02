import re 
import os
import time
import json
import requests
import mimetypes
import google.generativeai as genai
# from load_creds import load_creds
from io import BytesIO
from pdfminer.high_level import extract_text
import pdfplumber
from docx import Document
from vertexai.generative_models import GenerativeModel
import vertexai
from dotenv import load_dotenv

load_dotenv()
vertexai.init(project="ringed-spirit-422415-v6", location="us-central1")

# Now read them from the environment
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

model = GenerativeModel(
    model_name="projects/558798320545/locations/us-central1/endpoints/7445388067461922816", 
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
    """Extracts score ranges, components, and coefficients from the DESCRIPTION text."""

    # Match total score range, requiring it to end with '.' or newline
    total_score_match = re.search(r"Total\s+Scores?\s+range:\s*(\d+)\s*-\s*(\d+)[\.]", description_text, re.IGNORECASE)
    if not total_score_match:
        total_score_match = re.search(r"Scores?\s+range:\s*(\d+)\s*-\s*(\d+)[\.]", description_text, re.IGNORECASE)

    # Match components list (ends with '.' or newline)
    components_match = re.search(r"Components?:\s*([^\.]+)", description_text, re.IGNORECASE)
    
    # Match component score range, ending with '.' or newline
    comp_score_match = re.search(r"Components?\s+Scores?\s+range:\s*(\d+)\s*-\s*(\d+)[\.]", description_text, re.IGNORECASE)

    # Match coefficients (ends with '.' or newline)
    coefficients_match = re.search(r"Coefficients?:\s*([^\.]+)", description_text, re.IGNORECASE)

    # Extract total score range
    if total_score_match:
        min_total, max_total = map(int, total_score_match.groups())
    else:
        min_total, max_total = 0, 0

    # Extract component score range
    if comp_score_match and components_match:
        min_comp, max_comp = map(int, comp_score_match.groups())
    else:
        min_comp, max_comp = 0, 0

    # Extract components
    components = []
    if components_match:
        components_text = components_match.group(1)
        components = [comp.strip() for comp in components_text.split(',') if comp.strip()]

    # Extract coefficients
    coefficients = []
    if coefficients_match and comp_score_match and components_match:
        coefficients_text = coefficients_match.group(1)
        coefficients = list(map(int, filter(None, re.split(r',\s*', coefficients_text))))

    print(f"Total Score Range: {min_total}-{max_total}, Component Score Range: {min_comp}-{max_comp}, Components: {components}, {coefficients}")
    return (min_total, max_total), (min_comp, max_comp), components, coefficients

def clean_feedback(feedback):
    """Cleans feedback by removing inline numbers and fixing abrupt endings."""
    # Remove markdown-style bold (**text** or __text__)
    feedback = re.sub(r"(\*\*|__)(.*?)\1", r"\2", feedback)

    # Remove inline numbers (like 'example1', 'point2', but not section numbers like 1. Introduction)
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)

    return feedback.strip()

def extract_score(response_text, min_total, max_total, min_comp, max_comp, components, coefficients):
    """Extracts overall score, individual component scores and feedback."""
    score = min_total
    component_scores = [min_comp] * len(components)
    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+(\.\d+)?", line)
            if match:
                score = min(max_total, max(min_total, int(match.group(0))))
        for comp in components:
            pattern = re.compile(rf"{re.escape(comp)}\s*:\s*(\d+(\.\d+)?)", re.IGNORECASE)
            match = pattern.search(response_text)
            if match:
                score_val = float(match.group(1))
                if score_val.is_integer():
                    score_val = int(score_val)
                component_scores[components.index(comp)] = min(max_comp, max(min_comp, score_val))

    if components:
        if coefficients:
            score = sum(coeff * score for coeff, score in zip(coefficients, component_scores))
        else:
            score = sum(component_scores)
    return score, component_scores


def evaluate_submissions(data, output_json_path=None):
    # with open(input_json_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    filepath = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if filepath and not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)  # Convert to absolute path if not already
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = filepath
    
    results = []
    
    description_urls = data.get("descriptions", [])
    description_content = "\n".join([load_file_content(url) for url in description_urls])
    (min_total, max_total), (min_comp, max_comp), components, coefficients = extract_score_ranges_and_components(description_content)
    # rubric_content = "\n".join([load_file_content(url) for entry in data.get("rubrics", []) for url in entry.get("rubric_urls", [])])
    
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

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    score_rp = model.generate_content(prompt)
                    score_text = score_rp.text.strip()
                    # if not score_text:
                    #     print(f"⚠️ Empty response for essay_id {essay_id}. Retrying... (Attempt {attempt+1}/{max_retries})")
                    #     time.sleep(10)  # Add a short delay before retrying
                    #     continue  # Skip the rest of the loop and retry
                    
                    # print(f"Response for essay_id {essay_id}:\n{score_text}")
                    break  # Break out if successful
                except Exception as e:
                    err_msg = str(e)
                    if "429 Resource exhausted" in err_msg:
                        # print(f"⚠️ Essay {essay_id}: {err_msg}")
                        # print(f"Waiting 60 seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(60)
                    else:
                        raise  # For any other error, do not retry
            
            score, scores = extract_score(
                score_text, min_total, max_total, min_comp, max_comp, components, coefficients
            )

            feedback_rp = feedback_model.generate_content(fb_prompt)
            feedback_text = feedback_rp.text.strip()
            # print(score_text)
            # print(feedback_text)

            feedback = clean_feedback(feedback_text)
            
            results.append({
                "submission_id": submission_id,
                "ovr": score,
                "scores": scores,   # array of component scores
                "components": components,
                "coefficients": coefficients,
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing submission_id {submission_id}: {e}")
            results.append({
                "submission_id": submission_id,
                "ovr": min_total,
                "scores": [min_comp] * len(components),    # array of component scores
                "components": components,
                "coefficients": coefficients,
                "feedback": "Error processing submission."
            })
    
    # print("--------------------")
    # print(prompt)
    # print(f"Evaluation complete. Results saved")
    return {"results": results}
    
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=4)