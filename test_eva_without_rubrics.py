import os
import pandas as pd
import google.generativeai as genai
import chardet
from load_creds import load_creds

creds = load_creds()

genai.configure(credentials=creds)

generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="tunedModels/aesmodeltest2-yt4me2y47joq",
  generation_config=generation_config,
)

# genai.update_tuned_model('tunedModels/aes-test-zp3k24er2uve', {"description":"This is my model."})


# Function to clean Gemini's response text
def clean_response(response):
    return response.replace("**", "").replace("  ", " ")


# Function to create a prompt for the Gemini API
def make_prompt(essay_text):
    escaped = essay_text.replace("\n", " ").replace("'", "").replace('"', "")
    prompt = (
        f"PROMPT: Evaluate the following answer with scores ranging from 1(lowest) to 6(highest) and the ouput contains only 1 score. The score should reflect the essay's content, organization, language use, and mechanics."
        f"ESSAY: '{escaped}'\n"
        # f"OUTPUT REQUIREMENTS:\n"
        # f"1. Provide a score from 1 to 6.\n"
        # f"2. Provide a brief explanation of the score, highlighting strengths and weaknesses."
    )
    return prompt


# Function to evaluate essays using the Gemini API
def evaluate_essays(input_csv_path, output_csv_path, num_rows=100):
    # Detect encoding
    with open(input_csv_path, 'rb') as f:
        result = chardet.detect(f.read())
        detected_encoding = result['encoding']
    print(f"Detected encoding: {detected_encoding}")

    # Read CSV using detected encoding and limit to `num_rows`
    df = pd.read_csv(input_csv_path, encoding=detected_encoding).head(num_rows)

    # Check for required columns
    required_columns = ['essay_id', 'full_text', 'score']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain the following columns: {required_columns}")

    # Add a column for predicted scores
    df['gemini_score'] = None
    df['evaluation'] = None  # Detailed evaluation

    # Process each essay
    for idx, row in df.iterrows():
        essay_text = row['full_text']
        try:
            prompt = make_prompt(essay_text)
            response = model.generate_content(prompt)
            response_text = clean_response(response.text)

            # Extract the predicted score (assume score is first line of response)
            gemini_score = response_text.split('\n')[0].strip()

            # Update the DataFrame with Gemini's score and evaluation
            df.at[idx, 'gemini_score'] = gemini_score
            df.at[idx, 'evaluation'] = response_text
        except Exception as e:
            print(f"Error processing essay_id {row['essay_id']}: {e}")
            df.at[idx, 'gemini_score'] = 'Error'
            df.at[idx, 'evaluation'] = str(e)

    # Save the results to a new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Evaluation complete. Results saved to {output_csv_path}")


# Example usage
input_csv = "train.csv"
output_csv = "train_with_gemini_scores1.csv"
evaluate_essays(input_csv, output_csv, num_rows=200)
