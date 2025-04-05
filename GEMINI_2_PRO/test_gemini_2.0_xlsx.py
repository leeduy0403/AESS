import re
import os
import time
import mimetypes
import google.generativeai as genai
import pandas as pd
from load_creds import load_creds
from io import BytesIO
from pdfminer.high_level import extract_text
from docx import Document

def extract_docx_text(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {e}"

genai.configure(credentials=load_creds())

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="models/gemini-2.0-pro-exp-02-05",
    generation_config=generation_config,
)

def clean_feedback(feedback):
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()

def extract_scores_and_feedback(response_text):
    coherence, lexical, grammar = 0, 0, 0
    feedback = ""

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("coherence and cohesion:"):
            match = re.search(r"\d+", line)
            if match:
                coherence = int(match.group(0))
        elif line.lower().startswith("lexical resource:"):
            match = re.search(r"\d+", line)
            if match:
                lexical = int(match.group(0))
        elif line.lower().startswith("grammatical range and accuracy:"):
            match = re.search(r"\d+", line)
            if match:
                grammar = int(match.group(0))
        elif line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()

    feedback = clean_feedback(feedback)
    ovr = min(6, max(1, coherence + lexical + grammar))
    return ovr, [coherence, lexical, grammar], feedback

def evaluate_essays(description_path, content_path, output_path):
    description_content = extract_docx_text(description_path)
    
    df = pd.read_excel(content_path)
    # df = df.head(1000)  # Limit to first 300 rows
    
    request_count = 0
    results = []
    
    for index, row in df.iterrows():
        if request_count >= 4:
            time.sleep(60)
            request_count = 0
        
        essay_id = row["essay_id"]
        essay_content = row["essay"]
        
        try:
            prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {essay_content}\n\n"
                f"PROMPT: Provide an evaluation of the given CONTENT based on the DESCRIPTION and RUBRIC. "
                f"Your response should follow this format exactly:\n\n"
                f"Score: \n"
                f"Coherence and Cohesion: \n"
                f"Lexical Resource: \n"
                f"Grammatical Range and Accuracy: \n"
                f"Feedback: (A clear and concise textual feedback, without any embedded numbers or scores.)"
            )

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            score, scores, feedback = extract_scores_and_feedback(response_text)

            results.append({
                "essay_id": essay_id,
                "ovr": score,
                "scores": scores,
                "components": [
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                ],
                "feedback": feedback
            })
            request_count += 1
        except Exception as e:
            print(f"Error processing essay_id {essay_id}: {e}")
            results.append({
                "essay_id": essay_id,
                "ovr": 0,
                "scores": [0, 0, 0],
                "components": [
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                ],
                "feedback": "Error processing submission."
            })
    
    results_df = pd.DataFrame(results)
    final_df = df.merge(results_df, on="essay_id", how="left")
    final_df.to_excel(output_path, index=False)
    
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_essays("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1_description.docx", "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1.xlsx", "output_gemini2.0_new.xlsx")
