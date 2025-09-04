import re

def extract_score(response_text, min_total, max_total, min_comp, max_comp, components, coefficients):
    score = min_total
    component_scores = [min_comp] * len(components)

    for comp in components:
        pattern = re.compile(rf"{re.escape(comp)}\s*:\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
        match = pattern.search(response_text)
        if match:
            val = float(match.group(1).replace(",", "."))
            component_scores[components.index(comp)] = min(max_comp, max(min_comp, val))

    if components:
        if coefficients:
            score = sum(c * s for c, s in zip(coefficients, component_scores))
        else:
            score = sum(component_scores)

    return score, component_scores