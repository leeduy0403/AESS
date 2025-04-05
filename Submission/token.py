import pandas as pd
import nltk

# Ensure necessary tokenizer is available
nltk.download("punkt")

def count_tokens_in_excel(file_path, text_column):
    """
    Counts the total number of tokens in a specified text column of an Excel file.
    
    :param file_path: Path to the Excel file.
    :param text_column: Name of the column containing text.
    :return: Total token count.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)

        # Check if the specified column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the file.")

        # Tokenize and count tokens in each row
        df["token_count"] = df[text_column].astype(str).apply(lambda x: len(nltk.word_tokenize(x)))

        # Return the total token count
        return df["token_count"].sum()
    
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
file_path = "training_set_rel3.xlsx"  # Replace with your actual file path
text_column = "essay"         # Replace with the column name containing text
total_tokens = count_tokens_in_excel(file_path, text_column)
print(f"Total token count: {total_tokens}")
