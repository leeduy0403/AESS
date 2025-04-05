import joblib
import pandas as pd

# Load Trained Model & Vectorizer
# reg_model = joblib.load("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//score_regression_model.pkl")
# vectorizer = joblib.load("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//tfidf_vectorizer.pkl")
reg_model = joblib.load("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//rf_regression_model.pkl")
vectorizer = joblib.load("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//rf_tfidf_vectorizer.pkl")

# Load Validation Dataset
valid_file_path = "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//valid_set_1.xlsx"
df_valid = pd.read_excel(valid_file_path)

# Ensure the "essay" column exists
if "essay" not in df_valid.columns:
    raise ValueError("Column 'essay' not found in the validation dataset.")

# Function to Predict Scores
def predict_score(essay_text):
    """Predict the overall score for a given essay."""
    essay_vector = vectorizer.transform([essay_text]).toarray()
    predicted_score = reg_model.predict(essay_vector)[0]
    return round(predicted_score) 

# Apply Predictions to Each Essay
df_valid["predicted_score"] = df_valid["essay"].apply(predict_score)

# Save Results to a New Excel File
output_file_path = "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//valid_set_with_predictions_rf.xlsx"
df_valid.to_excel(output_file_path, index=False)

print(f"Predictions saved successfully to {output_file_path}")
