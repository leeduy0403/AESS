import re

def normalize_number(text):
    return float(text.strip().replace(',', '.').rstrip('.'))

def clean_feedback(feedback):
    feedback = re.sub(r"(\*\*|__)(.*?)\1", r"\2", feedback)
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()