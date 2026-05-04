import re
from pathlib import Path
import pandas as pd


def normalize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_taxonomy_terms(reference_folder: Path) -> set[str]:
    terms = set()

    if not reference_folder.exists():
        return terms

    bad_terms = {
        "skill", "skills", "knowledge", "abilities", "ability", "title", "description",
        "occupation", "occupations", "task", "tasks", "category", "year", "years",
        "information", "data", "element", "elements", "activity", "activities",
        "science", "design", "leadership", "marketing", "project management"
    }

    for file_path in list(reference_folder.glob("*.xlsx")) + list(reference_folder.glob("*.csv")):
        try:
            if file_path.suffix.lower() == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception:
            continue

        for column in df.columns:
            for value in df[column]:
                text = normalize_value(value)

                if not text:
                    continue
                if len(text) < 4 or len(text) > 40:
                    continue
                if text.isdigit():
                    continue
                if text in bad_terms:
                    continue
                if re.fullmatch(r"[\W_]+", text):
                    continue
                if re.search(r"\b\d+\b", text):
                    continue
                if text.count(" ") > 4:
                    continue

                terms.add(text)

    return terms


def extract_taxonomy_matches(text: str, taxonomy_terms: set[str]) -> list[str]:
    text_lower = text.lower()
    found = []

    for term in taxonomy_terms:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, text_lower):
            found.append(term)

    return sorted(set(found))