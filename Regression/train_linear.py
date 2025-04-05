import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
file_path = "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1.xlsx"
df = pd.read_excel(file_path)

# Step 2: Keep Relevant Columns
df = df[["essay", "domain1_score"]].dropna()

# Step 3: Convert Text to Numerical Features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_text = vectorizer.fit_transform(df["essay"])
X_text_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

# Target variable (overall score)
y = df["domain1_score"].values

# Step 4: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X_text_df, y, test_size=0.2, random_state=42)

# Step 5: Train a Linear Regression Model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Step 6: Evaluate Model Performance
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance: MSE = {mse:.2f}, R² Score = {r2:.2f}")

# Step 7: Save Model & Vectorizer for Future Use
joblib.dump(reg_model, "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//score_regression_model.pkl")
joblib.dump(vectorizer, "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
