import re 
import os
import time
import json
import requests 
import mimetypes
import pandas as pd
import google.generativeai as genai
# from load_creds import load_creds
from io import BytesIO
from pdfminer.high_level import extract_text
import pdfplumber
from docx import Document
from vertexai.generative_models import GenerativeModel
import vertexai
from dotenv import load_dotenv
from rest_framework import status

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

# def load_file_content(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         content_type = response.headers.get("Content-Type", "")
#         ext = mimetypes.guess_extension(content_type)
        
#         if ext in [".docx", ".doc"]:
#             return extract_docx_text(BytesIO(response.content))
#         elif ext == ".pdf":
#             return extract_pdf_text(BytesIO(response.content))
#         elif ext == ".txt":
#             return response.text
#         else:
#             return "Unsupported file format"
#     except Exception as e:
#         return f"Error retrieving file: {e}"

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
        elif ext in [".xlsx", ".xls"]:
            return ("excel", BytesIO(response.content))
        else:
            return None  # Unsupported format
    except Exception as e:
        return None  # Error retrieving file
    
def load_submission_file(url):
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
            return None  # Unsupported format
    except Exception as e:
        return None  # Error retrieving file

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

def normalize_number(text):
    """Converts a string with ',' or '.' to a float, and handles trailing dot like '2.'."""
    return float(text.strip().replace(',', '.').rstrip('.'))

def safe_float(value):
    if pd.isnull(value):
        return 0.0
    if isinstance(value, str):
        value = value.replace(',', '.')  # Replace comma with period
    return float(value)

def extract_score_ranges_and_components(description_text):
    """Extracts score ranges, components, component ranges, and coefficients from the DESCRIPTION text."""

    float_pattern = r"\d+(?:[.,]\d+)?"
    dash_pattern = r"[-–—]"  # matches -, –, or —

    # Match total score range
    total_score_match = re.search(
        rf"Total\s+Scores?\s+range:\s*({float_pattern})\s*{dash_pattern}\s*({float_pattern})",
        description_text,
        re.IGNORECASE
    )
    if not total_score_match:
        total_score_match = re.search(
            rf"Scores?\s+range:\s*({float_pattern})\s*{dash_pattern}\s*({float_pattern})",
            description_text,
            re.IGNORECASE
        )
        
    # Check again
    if not total_score_match:
        return None

    # Match component score range (general)
    comp_score_match = re.search(
        rf"Components?\s+Scores?\s+range:\s*({float_pattern})\s*{dash_pattern}\s*({float_pattern})", 
        description_text, 
        re.IGNORECASE
    )

    # Match coefficients
    coefficients_match = re.search(
        r"Coefficients?:\s*([0-9.,;\s]+)", description_text, re.IGNORECASE
    )
    if not coefficients_match:
        coefficients_match = re.search(
            r"Coefficient:\s*([0-9.,;\s]+)", description_text, re.IGNORECASE
        )

    # Extract total score range
    if total_score_match:
        min_total, max_total = map(normalize_number, total_score_match.groups())
    else:
        min_total, max_total = 0.0, 0.0

    # Extract components and component ranges
    components = []
    component_ranges = []

    # Match patterns like: Ideas and Content (0–4), Organization (0 - 5), ...
    comp_pattern = re.compile(
        rf"([\w\s&/]+?)\s*\(\s*({float_pattern})\s*{dash_pattern}\s*({float_pattern})\s*\)[.,]?", 
        re.UNICODE
    )
    comp_matches = comp_pattern.findall(description_text)
    if comp_matches:
        for name, min_c, max_c in comp_matches:
            components.append(name.strip())
            component_ranges.append((normalize_number(min_c), normalize_number(max_c)))
    else:
        # Fallback to plain component names
        components_match = re.search(r"Components?:\s*([^\.]+)", description_text, re.IGNORECASE)
        if components_match:
            components_text = components_match.group(1)
            components = [comp.strip() for comp in components_text.split(',') if comp.strip()]

        if components:
            # Extract general component score range
            if comp_score_match:
                min_comp, max_comp = map(normalize_number, comp_score_match.groups())
                for _ in components:
                    component_ranges.append((min_comp, max_comp))
            else:
                # Try to extract individual ranges
                for comp in components:
                    match = re.search(
                        rf"{re.escape(comp)}\s*\(\s*({float_pattern})\s*{dash_pattern}\s*({float_pattern})\s*\)", 
                        description_text, 
                        re.IGNORECASE
                    )
                    if match:
                        min_c, max_c = map(normalize_number, match.groups())
                        component_ranges.append((min_c, max_c))
                    else:
                        return None

    # Extract coefficients
    coefficients = []
    if coefficients_match:
        coefficients_text = coefficients_match.group(1)
        coefficients = [
            normalize_number(coef) for coef in re.split(r'[,;\s]+', coefficients_text) if coef.strip()
        ]

    # Mismatch check
    if components and coefficients and len(components) != len(coefficients):
        print(f"⚠️ Warning: {len(components)} components but {len(coefficients)} coefficients found!")
        return None

    # print(f"Total Score Range: {min_total}-{max_total}, Components: {components}, Component Ranges: {component_ranges}, Coefficients: {coefficients}")
    return {"scores":(min_total, max_total), "component_names": components, "component_ranges": component_ranges, "coefficients": coefficients}

def extract_rubric_from_excel(excel_path):
    # Read the only sheet
    df = pd.read_excel(excel_path)

    # Normalize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Check if "Coefficient" column exists, if not, create it with all values set to 1
    if "coefficient" not in df.columns or df["coefficient"].isnull().all():
        df["coefficient"] = 1
        
    # Check if "Coefficient" column exists, if not, create it with all values set to 1
    if "min score" not in df.columns or df["min score"].isnull().all():
        df["min score"] = 0
    
    # Check if required columns exist
    required_columns = ["component", "max score", "min score", "coefficient"]
    for col in required_columns:
        if col not in df.columns:
            return None
    # Check if "Component" column is empty
    if df["component"].isnull().all():
        return None
    # Check if "Min Score" and "Max Score" columns are empty
    if df["max score"].isnull().all():
        return None
    # Check if "Component" column has duplicate values
    if df["component"].duplicated().any():
        return None
    # Check if "Min Score" and "Max Score" columns have non-numeric values
    if not pd.api.types.is_numeric_dtype(df["min score"]) or not pd.api.types.is_numeric_dtype(df["max score"]):
        return None
    # Check if "Coefficient" column has non-numeric values
    if not pd.api.types.is_numeric_dtype(df["coefficient"]):
        return None
    # Check if "Min Score" and "Max Score" columns have negative values
    if any(df["min score"] < 0) or any(df["max score"] < 0):
        return None
    # Check if "Min Score" is less than "Max Score"
    if any(df["min score"] >= df["max score"]):
        return None
    # Check if "Coefficient" column has negative values
    if any(df["coefficient"] < 0):
        return None

    components = []
    component_names = []
    component_ranges = []
    coefficients = []


    for _, row in df.iterrows():
        component = {
            "name": row["component"],
            "coefficient": safe_float(row["coefficient"]),
            "min_score": safe_float(row["min score"]),
            "max_score": safe_float(row["max score"])   
        }
        components.append(component)
        component_names.append(component["name"])
        component_ranges.append((component["min_score"], component["max_score"]))
        coefficients.append(component["coefficient"])

    # Calculate min and max total score
    min_total_score = sum(c["min_score"] * c["coefficient"] for c in components)
    max_total_score = sum(c["max_score"] * c["coefficient"] for c in components)

    # print(f"Total Score Range: {min_total_score}-{max_total_score}, Components: {component_names}, Component Ranges: {component_ranges}, Coefficients: {coefficients}")

    return {"scores":(min_total_score, max_total_score), "component_names": component_names, "component_ranges": component_ranges, "coefficients": coefficients}

def clean_feedback(feedback):
    """Cleans feedback by removing inline numbers and fixing abrupt endings."""
    # Remove markdown-style bold (**text** or __text__)
    feedback = re.sub(r"(\*\*|__)(.*?)\1", r"\2", feedback)

    # Remove inline numbers (like 'example1', 'point2', but not section numbers like 1. Introduction)
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)

    return feedback.strip()

def extract_score(response_text, min_total, max_total, components, component_ranges, coefficients=None):
    """Extracts overall score, individual component scores and feedback."""

    # Initialize with defaults
    score = min_total
    component_scores = []

    # Extract each component score
    for i, comp in enumerate(components):
        min_c, max_c = component_ranges[i]
        pattern = re.compile(rf"{re.escape(comp)}\s*:\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
        match = pattern.search(response_text)
        if match:
            score_val = float(match.group(1).replace(',', '.'))
            score_val = min(max_c, max(min_c, score_val))  # Clamp within range
        else:
            score_val = min_c  # Default to min if missing
        component_scores.append(score_val)

    # Extract total score
    match = re.search(r"Score\s*:\s*(\d+(?:[.,]\d+)?)", response_text, re.IGNORECASE)
    if match:
        score_val = float(match.group(1).replace(',', '.'))
        score = min(max_total, max(min_total, score_val))

    if components:
        # If total score missing, calculate from components
        if coefficients:
            score = sum(coeff * comp_score for coeff, comp_score in zip(coefficients, component_scores))
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
    # description_content = "\n".join([load_file_content(url) for url in description_urls])
    # (min_total, max_total), (min_comp, max_comp), components, coefficients = extract_score_ranges_and_components(description_content)
    # Loop through URLs to find and process files
    description_content_list = []
    rubric_extracted = False

    for url in description_urls:
        content = load_file_content(url)
        if content is None:
            return {"result": "Invalid file format", "status": status.HTTP_400_BAD_REQUEST}   # return BAD REQUEST
        if isinstance(content, tuple) and content[0] == "excel":
            # Found Excel, extract rubric directly
            results = extract_rubric_from_excel(content[1])
            if results is None:
                return {"result": "Invalid Excel column names or values", "status": status.HTTP_400_BAD_REQUEST}
            else:
                (min_total, max_total) = results["scores"]
                components = results["component_names"]
                component_ranges = results["component_ranges"]
                coefficients = results["coefficients"]
                rubric_extracted = True
                
        else:
            description_content_list.append(content)

    description_content = "\n".join(description_content_list)

    if not rubric_extracted:
        # If no Excel, fallback to extract from description content
        # (min_total, max_total), components, component_ranges, coefficients = extract_score_ranges_and_components(description_content)
        results = extract_score_ranges_and_components(description_content)
        if results is None:
            return {"result": "Invalid Document format", "status": status.HTTP_400_BAD_REQUEST}
        else:
            (min_total, max_total) = results["scores"]
            components = results["component_names"]
            component_ranges = results["component_ranges"]
            coefficients = results["coefficients"]

    for submission in data.get("submissions", []):
        submission_id = submission["submission_id"]
        submission_urls = submission["submission_urls"]
        
        submission_texts = []
        for url in submission_urls:
            content = load_submission_file(url)
            if content is None:
                return {"result": "Invalid file format", "status": status.HTTP_400_BAD_REQUEST}
            else:
                submission_texts.append(content)
            
        submission_content = "\n".join(submission_texts)
        
        try:
            # Scoring prompt
            prompt = (
                f"Evaluate the given CONTENT based on the DESCRIPTION. If CONTENT does not relevant to DESCRIPTION, score will be minimum. For each essay, provide the following:\n"
                f"1. An overall score.\n"
                f"2. A score for each component(if DESCRIPTION has).\n"
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"Follow the exact format below in your response:\n\n"
                f"Score: (a number from {min_total}-{max_total})\n"
            )
            for i, comp in enumerate(components):
                min_c, max_c = component_ranges[i]
                prompt += f"{comp}: (a number from {min_c}-{max_c})\n"
            
            prompt += f"Example:\nScore: {min_total}\n"
            for i, comp in enumerate(components):
                min_c, max_c = component_ranges[i]
                prompt += f"{comp}: {min_c}\n"

            # Feedback prompt
            fb_prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {submission_content}\n\n"
                f"Your task: Provide **only constructive feedback** on the CONTENT, based on the DESCRIPTION. Check carefully If CONTENT does not relevant to DESCRIPTION, Overall Feedback will show 'The content is not related to the description' and explain. Please give feedback as many details as possible\n"
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
                    score_rp = feedback_model.generate_content(prompt)
                    score_text = score_rp.text.strip()
                    if not score_text:
                        print(f"⚠️ Empty response. Retrying... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(10)  # Add a short delay before retrying
                        continue  # Skip the rest of the loop and retry
                    else:
                        break # Break out if successful
                    
                    # print(f"Response for essay_id {essay_id}:\n{score_text}")
                except Exception as e:
                    err_msg = str(e)
                    if "429 Resource exhausted" in err_msg:
                        # print(f"⚠️ Essay {essay_id}: {err_msg}")
                        # print(f"Waiting 60 seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(60)
                    else:
                        raise  # For any other error, do not retry
            
            score, scores = extract_score(score_text, min_total, max_total, components, component_ranges, coefficients)

            feedback_rp = feedback_model.generate_content(fb_prompt)
            feedback_text = feedback_rp.text.strip()

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
                "scores": [0] * len(components),    # array of component scores
                "components": components,
                "coefficients": coefficients,
                "feedback": "Error processing submission."
            })
    
    # print("--------------------")
    # print(prompt)
    # print(f"Evaluation complete. Results saved")
    return {"results": results, "status": status.HTTP_200_OK}  # return OK
    
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=4)