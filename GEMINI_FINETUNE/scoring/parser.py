import re
from data_utils.text_utils import normalize_number

def extract_score_ranges_and_components(description_text):
    float_pattern = r"\d+(?:[.,]\d+)?\.?"

    total_score_match = re.search(
        rf"Total\s+Scores?\s+range:\s*({float_pattern})\s*-\s*({float_pattern})[\.]", 
        description_text, re.IGNORECASE
    ) or re.search(
        rf"Scores?\s+range:\s*({float_pattern})\s*-\s*({float_pattern})[\.]", 
        description_text, re.IGNORECASE
    )

    comp_score_match = re.search(
        rf"Components?\s+Scores?\s+range:\s*({float_pattern})\s*-\s*({float_pattern})[\.]", 
        description_text, re.IGNORECASE
    )

    components_match = re.search(r"Components?:\s*([^\.]+)", description_text, re.IGNORECASE)

    coefficients_match = re.search(r"Coefficients?:\s*([0-9.,;\s]+)", description_text, re.IGNORECASE) \
        or re.search(r"Coefficient:\s*([0-9.,;\s]+)", description_text, re.IGNORECASE)

    if total_score_match:
        min_total, max_total = map(normalize_number, total_score_match.groups())
    else:
        min_total, max_total = 0.0, 0.0

    if comp_score_match:
        min_comp, max_comp = map(normalize_number, comp_score_match.groups())
    else:
        min_comp, max_comp = 0.0, 0.0

    components = []
    if components_match:
        components = [c.strip() for c in components_match.group(1).split(",") if c.strip()]

    coefficients = []
    if coefficients_match:
        coefficients = [normalize_number(c) for c in re.split(r";\s*", coefficients_match.group(1)) if c.strip()]

    return (min_total, max_total), (min_comp, max_comp), components, coefficients