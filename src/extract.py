import re
import html
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


COMMON_SKILLS = [
    "python", "java", "sql", "aws", "azure", "docker",
    "kubernetes", "terraform", "linux", "git", "devops",
    "machine learning", "excel", "tableau", "power bi",
    "javascript", "react", "node.js", "c++", "pandas",
    "customer service", "project management", "communication",
    "sales", "marketing", "data analysis", "leadership",
    "teamwork", "problem solving", "training", "analytics",
    "microsoft office", "word", "powerpoint", "hotel management",
    "hospitality", "front office", "budgeting", "forecasting"
]


def strip_html_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str):
    text = strip_html_tags(text)
    text_lower = text.lower()
    found = []

    for skill in COMMON_SKILLS:
        if skill in text_lower:
            found.append(skill)

    return sorted(list(set(found)))


def extract_email(text: str):
    text = strip_html_tags(text)
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None


def extract_phone(text: str):
    text = strip_html_tags(text)
    match = re.search(r'(\+?\d[\d\-\(\) ]{8,}\d)', text)
    return match.group(0) if match else None


def extract_entities(text: str):
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }

    if nlp is None:
        return entities

    clean_text = strip_html_tags(text)
    doc = nlp(clean_text)

    for ent in doc.ents:
        if ent.label_ in entities:
            value = ent.text.strip()

            if len(value) < 2:
                continue
            if "<" in value or ">" in value:
                continue
            if "class=" in value or "id=" in value:
                continue

            entities[ent.label_].append(value)

    for key in entities:
        entities[key] = sorted(list(set(entities[key])))

    return entities


def extract_years_experience(text: str):
    text = strip_html_tags(text).lower()

    matches = re.findall(r'(\d+)\+?\s+years?', text)
    if matches:
        return max(int(x) for x in matches)

    return 0


def extract_all(text: str):
    clean_text = strip_html_tags(text)

    return {
        "email": extract_email(clean_text),
        "phone": extract_phone(clean_text),
        "skills": extract_skills(clean_text),
        "experience_years": extract_years_experience(clean_text),
        "entities": extract_entities(clean_text)
    }