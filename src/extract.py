# src/extract.py

import re
import spacy

# Load small model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise Exception("Run: python -m spacy download en_core_web_sm")


COMMON_SKILLS = [
    "python", "java", "sql", "aws", "azure", "docker",
    "kubernetes", "terraform", "linux", "git", "devops",
    "machine learning", "excel", "tableau", "power bi",
    "javascript", "react", "node.js", "c++", "pandas"
]


def extract_skills(text: str):
    """
    Find skills using keyword dictionary
    """
    text_lower = text.lower()
    found = []

    for skill in COMMON_SKILLS:
        if skill in text_lower:
            found.append(skill)

    return sorted(list(set(found)))


def extract_email(text: str):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None


def extract_phone(text: str):
    match = re.search(r'(\+?\d[\d\-\(\) ]{8,}\d)', text)
    return match.group(0) if match else None


def extract_entities(text: str):
    """
    Uses spaCy for names/orgs/education/company
    """
    doc = nlp(text)

    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities


def extract_years_experience(text: str):
    match = re.findall(r'(\d+)\+?\s+years?', text.lower())
    if match:
        return max([int(x) for x in match])
    return 0


def extract_all(text: str):
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_years_experience(text),
        "entities": extract_entities(text)
    }