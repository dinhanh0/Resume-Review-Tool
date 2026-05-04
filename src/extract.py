import re
import html
from datetime import datetime

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


COMMON_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    ".net",
    ".net core",
    ".net framework",
    "asp.net",
    "asp.net mvc",
    "blazor",
    "react",
    "node.js",
    "node",
    "html",
    "css",
    "sql",
    "ms sql",
    "mssql",
    "mysql",
    "postgresql",
    "oracle",
    "api",
    "api development",
    "rest api",
    "git",
    "github",
    "ci/cd",
    "ci cd",
    "docker",
    "kubernetes",
    "linux",
    "aws",
    "azure",
    "excel",
    "microsoft office",
    "word",
    "powerpoint",
    "pandas",
    "machine learning",
    "data analysis",
    "data structures",
    "algorithms",
    "software engineering",
    "full stack",
    "full-stack",
    "customer service",
    "communication",
    "teamwork",
    "problem solving",
    "sales",
    "training",
    "hospitality",
    "front office",
    "budgeting",
    "forecasting",
]

VAGUE_SKILLS = {
    "science",
    "design",
    "leadership",
    "database",
    "databases",
    "marketing",
    "project management",
}

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


def strip_html_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = strip_html_tags(text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill_text(text: str) -> str:
    return normalize_text(text).lower()


def extract_skills(text: str):
    text_lower = normalize_skill_text(text)
    found = set()

    special_skills = {
        "c++", "c#", ".net", ".net core", ".net framework",
        "ci/cd", "node.js", "asp.net", "asp.net mvc"
    }

    for skill in COMMON_SKILLS:
        skill_lower = skill.lower()

        if skill_lower in special_skills:
            if skill_lower in text_lower:
                found.add(skill_lower)
            continue

        pattern = rf"\b{re.escape(skill_lower)}\b"
        if re.search(pattern, text_lower):
            found.add(skill_lower)

    found = {skill for skill in found if skill not in VAGUE_SKILLS}
    return sorted(found)


def extract_email(text: str):
    text = normalize_text(text)
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_phone(text: str):
    text = normalize_text(text)
    match = re.search(r"(\+?\d[\d\-\(\) ]{8,}\d)", text)
    return match.group(0) if match else None


def extract_entities(text: str):
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": [],
    }

    if nlp is None:
        return entities

    clean_text = normalize_text(text)
    doc = nlp(clean_text)

    blocked_terms = {
        "current company name",
        "city",
        "state",
        "company name",
        "present",
    }

    for ent in doc.ents:
        if ent.label_ not in entities:
            continue

        value = ent.text.strip()
        value_lower = value.lower()

        if len(value) < 2:
            continue
        if "<" in value or ">" in value:
            continue
        if "class=" in value_lower or "id=" in value_lower:
            continue
        if value_lower in blocked_terms:
            continue
        if len(value) > 60:
            continue

        entities[ent.label_].append(value)

    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities


def extract_explicit_years_experience(text: str) -> float:
    text_lower = normalize_text(text).lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s+years?", text_lower)

    if not matches:
        return 0.0

    return max(float(x) for x in matches)


def get_experience_section(text: str) -> str:
    clean_text = normalize_text(text)

    match = re.search(
        r"\bexperience\b(.*?)(?:\bclasses and projects\b|\beducation\b|\breferences\b|$)",
        clean_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1)

    return clean_text


def parse_month_year(month_text: str, year_text: str):
    month_num = MONTH_MAP.get(month_text.lower())
    if month_num is None:
        return None

    try:
        year_num = int(year_text)
    except ValueError:
        return None

    return datetime(year_num, month_num, 1)


def months_between(start_dt: datetime, end_dt: datetime) -> int:
    return (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)


def extract_date_ranges(text: str):
    experience_text = get_experience_section(text)

    pattern = re.compile(
        rf"\b({MONTH_PATTERN})\s+(\d{{4}})\s*-\s*(Present|Current|{MONTH_PATTERN})\s*(\d{{4}})?",
        flags=re.IGNORECASE,
    )

    ranges = []

    for match in pattern.finditer(experience_text):
        start_month, start_year, end_month, end_year = match.groups()

        start_dt = parse_month_year(start_month, start_year)
        if start_dt is None:
            continue

        if end_month.lower() in {"present", "current"}:
            now = datetime.now()
            end_dt = datetime(now.year, now.month, 1)
        else:
            if not end_year:
                continue
            end_dt = parse_month_year(end_month, end_year)
            if end_dt is None:
                continue

        if end_dt >= start_dt:
            ranges.append((start_dt, end_dt))

    return ranges


def merge_overlapping_ranges(ranges):
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]

    for current_start, current_end in ranges[1:]:
        last_start, last_end = merged[-1]

        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def extract_date_range_experience(text: str) -> float:
    ranges = extract_date_ranges(text)
    merged_ranges = merge_overlapping_ranges(ranges)

    total_months = 0
    for start_dt, end_dt in merged_ranges:
        total_months += months_between(start_dt, end_dt)

    years = total_months / 12.0
    return round(years, 1)


def extract_years_experience(text: str):
    explicit_years = extract_explicit_years_experience(text)
    ranged_years = extract_date_range_experience(text)

    best_years = max(explicit_years, ranged_years)

    if best_years == 0:
        return 0
    if float(best_years).is_integer():
        return int(best_years)

    return best_years


def extract_all(text: str):
    clean_text = normalize_text(text)

    return {
        "email": extract_email(clean_text),
        "phone": extract_phone(clean_text),
        "skills": extract_skills(clean_text),
        "experience_years": extract_years_experience(clean_text),
        "entities": extract_entities(clean_text),
    }