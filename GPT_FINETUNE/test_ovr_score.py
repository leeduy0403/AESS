import re
import os
import time
import mimetypes
# import google.generativeai as genai
import openai
import pandas as pd
# from load_creds import load_creds
from io import BytesIO
from pdfminer.high_level import extract_text
from docx import Document

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

def extract_docx_text(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {e}"

def extract_score_range(description_text):
    """Extracts the valid score range from the DESCRIPTION text."""
    match = re.search(r"Score\s+range:\s*(\d+)\s*-\s*(\d+)", description_text, re.IGNORECASE)
    if match:
        min_score, max_score = map(int, match.groups())
    else:
        min_score, max_score = 0, 6  # Default range if not found
    print(f"min_score: {min_score}, max_score: {max_score}")
    return min_score, max_score

# genai.configure(credentials=load_creds())

# generation_config = {
#     "temperature": 0.7,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#     model_name="tunedModels/aesmodeltest4-oioi4ubuqer6",
#     generation_config=generation_config,
# )

def clean_feedback(feedback):
    """Cleans feedback by removing inline numbers and fixing abrupt endings."""
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()

def extract_score_and_feedback(response_text, min_score, max_score):
    """Extracts only the overall score & feedback."""
    score, feedback = min_score, ""

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            match = re.search(r"\d+", line)
            if match:
                score = min(max_score, max(min_score, int(match.group(0))))  # Ensure valid range
        elif line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()

    feedback = clean_feedback(feedback)

    return score, feedback

def evaluate_essays(description_path, content_path, output_path):
    """Evaluates essays based on the extracted description and generates scores & feedback."""
    description_content = extract_docx_text(description_path)
    min_score, max_score = extract_score_range(description_content)  # Extract score range dynamically

    df = pd.read_excel(content_path)
    
    results = []
    
    for index, row in df.iterrows():
        essay_id = row["essay_id"]
        essay_content = row["essay"]
        
        try:
            # 0 shot prompt
            # prompt = (
            #     f"DESCRIPTION: {description_content}\n"
            #     f"CONTENT: {essay_content}\n\n"
            #     f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION. "
            #     f"Follow the exact format below in your response:\n\n"
            #     f"Score: (a number from {min_score}-{max_score})\n"
            #     f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)"
            # )

            # 1 shot prompt
            prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {essay_content}\n\n"
                f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION. "
                f"Follow the exact format below in your response:\n\n"
                f"Score: (a number from {min_score}-{max_score})\n"
                f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)"

                f"Example:\n"
                f"Score: 9\n"
                f"Feedback: The essay presents clear arguments and a strong structure. To enhance clarity, focus on smoother transitions and more precise word choices.\n\n"
            )

            # CoT prompt
            # prompt = (
            #     f"DESCRIPTION: {description_content}\n"
            #     f"CONTENT: {essay_content}\n\n"
            #     f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION. "
            #     f"Follow the exact format below in your response:\n\n"
            #     f"Score: (a number from {min_score}-{max_score})\n"
            #     f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)"

            #     f"Step 1: Identify the key strengths and weaknesses in the CONTENT.\n"
            #     f"Step 2: Analyze Coherence and Cohesion – How well are ideas structured and connected?\n"
            #     f"Step 3: Evaluate Lexical Resource – Does the vocabulary demonstrate variety and precision?\n"
            #     f"Step 4: Assess Grammatical Range and Accuracy – Are there significant errors, or is the writing mostly correct?\n"
            #     f"Step 5: Assign a score within the range of {min_score}-{max_score}.\n"
            #     f"Step 6: Provide final feedback that is clear, actionable, and free from embedded numbers or scores.\n\n"

            #     f"Example:\n"
            #     f"Scores: 9 \n"
            #     f"Feedback: The essay presents clear arguments and a strong structure. Focus on improving transitions and varying word choices to enhance clarity.\n\n"

            #     f"Now, follow this structured reasoning process and provide your evaluation."
            # )

            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",  # You can also use "gpt-3.5-turbo" for faster responses
                messages=[
                    {"role": "system", "content": "You are an essay evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()

            score, feedback = extract_score_and_feedback(response_text, min_score, max_score)

            results.append({
                "essay_id": essay_id,
                "ovr": score,  # ✅ Only overall score
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing essay_id {essay_id}: {e}")
            results.append({
                "essay_id": essay_id,
                "ovr": min_score,
                "feedback": "Error processing submission."
            })
    
    results_df = pd.DataFrame(results)
    final_df = df.merge(results_df, on="essay_id", how="left")
    final_df.to_excel(output_path, index=False)
    
    print(f"✅ Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_essays(
        "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1_description.docx",
        "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1.xlsx",
        "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//OUTPUT//one_shot//output_gpt4mini_set1.xlsx"
    )
    # evaluate_essays(
    #     "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET3//essay_set_3_description.docx",
    #     "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET3//essay_set_3.xlsx",
    #     "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//OUTPUT//CoT//output_gemini1.5_set3.xlsx"
    # )
