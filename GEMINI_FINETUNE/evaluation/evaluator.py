import time
import pandas as pd
from config.settings import model, feedback_model
from data_utils.file_loader import load_file_content, extract_pdf_text, extract_docx_text
from data_utils.text_utils import clean_feedback
from scoring.parser import extract_score_ranges_and_components
from scoring.extractor import extract_score

def evaluate_essays(description_path, content_path, output_path):
    if description_path.lower().startswith("http"):
        description_content = load_file_content(description_path)
    else:
        if description_path.lower().endswith(".pdf"):
            description_content = extract_pdf_text(description_path)
        elif description_path.lower().endswith(".docx"):
            description_content = extract_docx_text(description_path)
        else:
            with open(description_path, encoding="utf-8") as f:
                description_content = f.read()

    (min_total, max_total), (min_comp, max_comp), components, coefficients = \
        extract_score_ranges_and_components(description_content)

    df = pd.read_excel(content_path)
    df = df.head(1).reset_index(drop=True)
    results = []

    request_count = 0
    start_time = time.time()

    for index, row in df.iterrows():
        essay_id = row["essay_id"]
        essay_content = row["essay"]

        try:
            request_count += 1
            elapsed = time.time() - start_time
            if request_count > 100:
                if elapsed < 60:
                    wait_time = 60 - elapsed
                    print(f"⏳ Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 1
                start_time = time.time()

            prompt = (
                f"Evaluate the given CONTENT based on the DESCRIPTION. Provide:\n"
                f"Score: (a number from {min_total}-{max_total})\n"
            )
            for comp in components:
                prompt += f"{comp}: (a number from {min_comp}-{max_comp})\n"

            fb_prompt = (
                f"DESCRIPTION: {description_content}\n"
                f"CONTENT: {essay_content}\n\n"
                f"Provide constructive feedback only.\n"
                f"Overall Feedback:\n"
            )
            for comp in components:
                fb_prompt += f"{comp} Feedback:\n"

            score_rp = model.generate_content(prompt)
            score_text = score_rp.text.strip()
            score, scores = extract_score(score_text, min_total, max_total, min_comp, max_comp, components, coefficients)

            feedback_rp = feedback_model.generate_content(fb_prompt)
            feedback_text = feedback_rp.text.strip()
            feedback = clean_feedback(feedback_text)

            results.append({
                "essay_id": essay_id,
                "ovr": score,
                "scores": scores,
                "components": components,
                "coefficients": coefficients,
                "feedback": feedback
            })
        except Exception as e:
            print(f"Error processing essay_id {essay_id}: {e}")
            results.append({
                "essay_id": essay_id,
                "ovr": min_total,
                "scores": [min_comp] * len(components),
                "components": components,
                "coefficients": coefficients,
                "feedback": "Error processing submission."
            })

    results_df = pd.DataFrame(results)
    final_df = df.merge(results_df, on="essay_id", how="left")
    final_df.to_excel(output_path, index=False)
    print(f"✅ Evaluation complete. Results saved to {output_path}")