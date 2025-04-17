import re 
import os
import json
import requests
import mimetypes
import google.generativeai as genai
from load_creds import load_creds
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
    
def extract_score_ranges_and_components(description_text):
    """Extracts score ranges and components from the DESCRIPTION text."""
    total_score_match = re.search(r"Total\s+Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    comp_score_match = re.search(r"Components?\s+Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    components_section = re.search(r"Components:\s*(.*?)(?:\n\s*\n|\Z)", description_text, re.IGNORECASE | re.DOTALL)

    if total_score_match:
        min_total, max_total = map(int, total_score_match.groups())
    else:
        min_total, max_total = 1, 6

    if comp_score_match:
        min_comp, max_comp = map(int, comp_score_match.groups())
    else:
        min_comp, max_comp = 1, 6

    components = []
    if components_section:
        lines = components_section.group(1).splitlines()
        for line in lines:
            comp = re.sub(r"^[\s\-•\d\.o]+", "", line).strip()
            if comp:
                components.append(comp)

    if not components:
        components = ["Coherence and Cohesion", "Lexical Resource", "Grammatical Range and Accuracy"]

    print(f"Total Score Range: {min_total}-{max_total}, Component Score Range: {min_comp}-{max_comp}, Components: {components}")
    return (min_total, max_total), (min_comp, max_comp), components

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

def extract_score_and_feedback(response_text, min_total, max_total, min_comp, max_comp, components):
    """Extracts overall score, individual component scores and feedback."""
    score, feedback = min_total, ""
    component_scores = [min_comp] * len(components)

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+", line)
            if match:
                score = min(max_total, max(min_total, int(match.group(0))))
        for i, comp in enumerate(components):
            if line.lower().startswith(comp.lower() + ":"):
                match = re.search(r"\d+", line)
                if match:
                    component_scores[i] = min(max_comp, max(min_comp, int(match.group(0))))
        if line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()

    feedback = clean_feedback(feedback)

    # # Ensure score is average of components
    # avg_score = round(sum(component_scores) / len(component_scores))
    # score = avg_score

    return score, component_scores, feedback


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
            prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION.\n"
                f"Follow the exact format below in your response:\n\n"
                f"Score: (a number from {min_total}-{max_total})\n"
            )
            for comp in components:
                prompt += f"{comp}: (a number from {min_comp}-{max_comp})\n"

            prompt += (
                f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)\n\n"
                f"Example:\n"
                f"Score: {min_total}\n"
            )

            for comp in components:
                prompt += f"{comp}: {min_comp}\n"
                
            prompt += (
                f"Feedback: The essay presents clear arguments and a strong structure.\n"
            )

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            score, scores, feedback = extract_score_and_feedback(
                response_text, min_total, max_total, min_comp, max_comp, components
            )

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
                "feedback": feedback
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
