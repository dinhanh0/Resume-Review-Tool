import os
import sys
import re
import html
from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import preprocess_text
from extract import extract_all, extract_skills
from match import overall_match


def read_txt_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_docx_file(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_pdf_file(file_path: str) -> str:
    text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)


def load_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return read_txt_file(file_path)
    if extension == ".docx":
        return read_docx_file(file_path)
    if extension == ".pdf":
        return read_pdf_file(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def strip_html_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_placeholder_text(text: str) -> str:
    bad_phrases = [
        "current company name",
        "company name",
        "city",
        "state",
        "state company",
        "state education",
        "state business administration",
        "current company",
        "current title",
        "current position",
        "present",
        "n/a"
    ]

    cleaned = text

    for phrase in bad_phrases:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_resume_text(text: str) -> str:
    text = strip_html_tags(text)
    text = clean_placeholder_text(text)
    return text


def clean_job_text(text: str) -> str:
    text = strip_html_tags(text)
    text = clean_placeholder_text(text)
    return text


def load_reference_terms(reference_folder: Path) -> set[str]:
    terms = set()

    if not reference_folder.exists():
        return terms

    bad_terms = {
        "im", "lv", "level", "levels", "skill", "skills",
        "knowledge", "abilities", "ability", "title",
        "description", "work", "worker", "job", "jobs",
        "occupation", "occupations", "activity", "activities",
        "element", "elements", "task", "tasks", "category",
        "city", "state", "company", "name", "current company name",
        "current", "present", "year", "years", "information",
        "data", "worker characteristics", "occupation title"
    }

    for file_path in reference_folder.glob("*.xlsx"):
        try:
            df = pd.read_excel(file_path)
        except Exception:
            continue

        for column in df.columns:
            for value in df[column]:
                text = normalize_text(value).lower()

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


def extract_reference_terms_from_text(text: str, reference_terms: set[str]) -> list[str]:
    text_lower = text.lower()
    found = []

    for term in reference_terms:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, text_lower):
            found.append(term)

    return sorted(list(set(found)))


def combine_skills(base_skills: list[str], extra_skills: list[str]) -> list[str]:
    combined = set()

    for skill in base_skills:
        if skill and isinstance(skill, str):
            combined.add(skill.lower().strip())

    for skill in extra_skills:
        if skill and isinstance(skill, str):
            combined.add(skill.lower().strip())

    return sorted(list(combined))


def classify_match(score: float) -> str:
    if score >= 80:
        return "Strong Match"
    if score >= 60:
        return "Moderate Match"
    return "Low Match"


def generate_feedback(result: dict, resume_data: dict) -> list[str]:
    feedback = []

    if result["overall_score"] >= 80:
        feedback.append("Strong overall match for this job description.")
    elif result["overall_score"] >= 60:
        feedback.append("Moderate match. Your resume aligns with several key requirements.")
    else:
        feedback.append("Low match. Your resume may need more tailoring for this role.")

    if result["matched_skills"]:
        feedback.append("Matched skills: " + ", ".join(result["matched_skills"][:10]))

    if result["missing_skills"]:
        feedback.append(
            "Consider adding or emphasizing these missing skills: "
            + ", ".join(result["missing_skills"][:10])
        )
    else:
        feedback.append("No missing skills were detected from the extracted job keywords.")

    if result["text_similarity"] < 50:
        feedback.append("Try aligning your wording more closely with the job description.")

    if resume_data["experience_years"] == 0:
        feedback.append("No clear years-of-experience phrase was detected in the resume.")
    else:
        feedback.append(
            f"Detected about {resume_data['experience_years']} year(s) of experience."
        )

    return feedback


def preview_text(text: str, max_length: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."


def print_list(title: str, items: list[str]) -> None:
    print(f"\n{title}:")
    if items:
        for item in items:
            print(f"- {item}")
    else:
        print("None")


def print_extracted_summary(title: str, data: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Email: {data.get('email') or 'Not found'}")
    print(f"Phone: {data.get('phone') or 'Not found'}")
    print(f"Years of Experience Found: {data.get('experience_years', 0)}")

    skills = data.get("skills", [])
    print("Skills Found:")
    if skills:
        for skill in skills:
            print(f"- {skill}")
    else:
        print("None")


def print_results(
    resume_raw: str,
    job_raw: str,
    resume_data: dict,
    job_data: dict,
    result: dict
) -> None:
    print("\n" + "=" * 70)
    print("RESUME REVIEW RESULTS")
    print("=" * 70)

    print("\nRESUME PREVIEW:")
    print(preview_text(resume_raw))

    print("\nJOB DESCRIPTION PREVIEW:")
    print(preview_text(job_raw))

    print_extracted_summary("WHAT THE PROGRAM EXTRACTED FROM THE RESUME", resume_data)
    print_extracted_summary("WHAT THE PROGRAM EXTRACTED FROM THE JOB DESCRIPTION", job_data)

    print(f"\nMatch Level: {classify_match(result['overall_score'])}")
    print(f"Overall Match Score: {result['overall_score']}%")
    print(f"Text Similarity Score: {result['text_similarity']}%")
    print(f"Skill Match Score: {result['skill_score']}%")

    print_list("Matched Skills", result["matched_skills"])
    print_list("Missing Skills", result["missing_skills"])

    print("\nFeedback:")
    for item in generate_feedback(result, resume_data):
        print(f"- {item}")

    print("=" * 70)


def load_resume_from_csv(csv_path: Path, row_index: int) -> str:
    df = pd.read_csv(csv_path)

    if "Resume_html" not in df.columns:
        raise ValueError("Resume.csv does not contain a 'Resume_html' column.")

    if row_index < 0 or row_index >= len(df):
        raise IndexError("Resume row index is out of range.")

    resume_text = normalize_text(df.loc[row_index, "Resume_html"])
    return clean_resume_text(resume_text)


def load_job_from_csv(csv_path: Path, row_index: int) -> str:
    df = pd.read_csv(csv_path)

    if "job_description" not in df.columns:
        raise ValueError("training_data.csv does not contain a 'job_description' column.")

    if row_index < 0 or row_index >= len(df):
        raise IndexError("Job description row index is out of range.")

    job_text = normalize_text(df.loc[row_index, "job_description"])
    return clean_job_text(job_text)


def get_multiline_text(label: str) -> str:
    print(f"\nPaste the {label}. Type END on a new line when finished:")
    lines = []

    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    print("Resume Review Tool")
    print("Choose an input mode:")
    print("1. Enter file paths for resume and job description")
    print("2. Paste resume and job description text")
    print("3. Use sample data from Resume.csv and training_data.csv")

    choice = input("Enter 1, 2, or 3: ").strip()

    resume_raw = ""
    job_raw = ""

    try:
        if choice == "1":
            resume_path = input("Enter path to resume file: ").strip()
            job_path = input("Enter path to job description file: ").strip()

            if not os.path.exists(resume_path):
                print(f"Resume file not found: {resume_path}")
                return

            if not os.path.exists(job_path):
                print(f"Job description file not found: {job_path}")
                return

            resume_raw = clean_resume_text(load_text(resume_path))
            job_raw = clean_job_text(load_text(job_path))

        elif choice == "2":
            resume_raw = clean_resume_text(get_multiline_text("resume text"))
            job_raw = clean_job_text(get_multiline_text("job description text"))

        elif choice == "3":
            resume_csv_path = PROJECT_ROOT / "data" / "Resume data" / "Resume.csv"
            job_csv_path = PROJECT_ROOT / "data" / "Resume data" / "training_data.csv"

            resume_index = int(input("Enter resume row index: ").strip())
            job_index = int(input("Enter job description row index: ").strip())

            resume_raw = load_resume_from_csv(resume_csv_path, resume_index)
            job_raw = load_job_from_csv(job_csv_path, job_index)

        else:
            print("Invalid choice.")
            return

    except Exception as error:
        print(f"Input error: {error}")
        return

    if not resume_raw.strip():
        print("Resume input is empty.")
        return

    if not job_raw.strip():
        print("Job description input is empty.")
        return

    reference_folder = PROJECT_ROOT / "data" / "Job skill reference data"
    reference_terms = load_reference_terms(reference_folder)

    resume_clean = preprocess_text(resume_raw)
    job_clean = preprocess_text(job_raw)

    resume_data = extract_all(resume_raw)

    base_resume_skills = resume_data["skills"]
    base_job_skills = extract_skills(job_raw)

    reference_resume_skills = extract_reference_terms_from_text(resume_raw, reference_terms)
    reference_job_skills = extract_reference_terms_from_text(job_raw, reference_terms)

    final_resume_skills = combine_skills(base_resume_skills, reference_resume_skills)
    final_job_skills = combine_skills(base_job_skills, reference_job_skills)

    resume_data["skills"] = final_resume_skills
    job_data = {
        "email": None,
        "phone": None,
        "experience_years": 0,
        "skills": final_job_skills
    }

    result = overall_match(
        resume_clean,
        job_clean,
        final_resume_skills,
        final_job_skills
    )

    print_results(resume_raw, job_raw, resume_data, job_data, result)


if __name__ == "__main__":
    main()