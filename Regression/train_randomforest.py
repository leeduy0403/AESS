import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
file_path = "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET1//essay_set_1.xlsx"
df = pd.read_excel(file_path)

# Keep Only Relevant Columns
df = df[["essay", "domain1_score"]].dropna()

# Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_text = vectorizer.fit_transform(df["essay"])
X_text_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

# Target Variable (Final Score)
y = df["domain1_score"].values

# Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_text_df, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Random Forest Performance: MSE = {mse:.2f}, R² Score = {r2:.2f}")

# Save Model & Vectorizer for Future Use
joblib.dump(rf_model, "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//rf_regression_model.pkl")
joblib.dump(vectorizer, "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Regression//rf_tfidf_vectorizer.pkl")

print("Random Forest Model and Vectorizer Saved Successfully!")
