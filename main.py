import sys
import re
import html
from pathlib import Path

import pdfplumber
from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import preprocess_text
from extract import extract_all
from taxonomy import load_taxonomy_terms, extract_taxonomy_matches
from domain_classifier import predict_domain
from match import overall_match


BAD_FINAL_SKILLS = {
    "science", "design", "leadership", "database", "databases",
    "marketing", "project management", "fast-paced", "team-oriented",
    "rapport", "attentive service", "level", "levels", "writing",
    "environment", "construction laborers", "administrative", "materials",
    "models", "tools", "delivery"
}


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
        "n/a",
    ]

    cleaned = text
    for phrase in bad_phrases:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_resume_text(text: str) -> str:
    return clean_placeholder_text(strip_html_tags(text))


def clean_job_text(text: str) -> str:
    return clean_placeholder_text(strip_html_tags(text))


def read_txt_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def read_docx_file(file_path: Path) -> str:
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_pdf_file(file_path: Path) -> str:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def load_resume_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return read_txt_file(file_path)
    if ext == ".docx":
        return read_docx_file(file_path)
    if ext == ".pdf":
        return read_pdf_file(file_path)

    raise ValueError(
        f"Unsupported resume file type: {ext}. Please use TXT, PDF, or DOCX."
    )


def list_resume_files(folder_path: Path) -> list[Path]:
    supported = {".txt", ".pdf", ".docx"}

    if not folder_path.exists():
        return []

    files = [
        file_path for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported
    ]

    return sorted(files, key=lambda p: p.name.lower())


def choose_resume_from_folder(folder_path: Path) -> str:
    supported_files = list_resume_files(folder_path)

    all_files = []
    if folder_path.exists():
        all_files = [file_path for file_path in folder_path.iterdir() if file_path.is_file()]

    if not supported_files:
        image_files = [
            file_path for file_path in all_files
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        if image_files:
            raise ValueError(
                "Image resumes are not supported. Please use a TXT, PDF, or DOCX file in the 'resume input' folder."
            )

        raise FileNotFoundError(
            f"No supported resume files found in folder: {folder_path}"
        )

    print("\nResume files found in 'resume input':")
    for index, file_path in enumerate(supported_files, start=1):
        print(f"{index}. {file_path.name}")

    choice = input("Choose a resume file number: ").strip()

    if not choice.isdigit():
        raise ValueError("Please enter a valid number.")

    file_index = int(choice) - 1
    if file_index < 0 or file_index >= len(supported_files):
        raise IndexError("Selected file number is out of range.")

    selected_file = supported_files[file_index]
    print(f"Using resume file: {selected_file.name}")

    return clean_resume_text(load_resume_file(selected_file))


def combine_skills(base_skills: list[str], extra_skills: list[str]) -> list[str]:
    combined = set()

    for skill in base_skills:
        if skill and isinstance(skill, str):
            combined.add(skill.lower().strip())

    for skill in extra_skills:
        if skill and isinstance(skill, str):
            combined.add(skill.lower().strip())

    return sorted(combined)


def clean_final_skills(skills: list[str]) -> list[str]:
    cleaned = []
    for skill in skills:
        skill = skill.lower().strip()
        if not skill:
            continue
        if skill in BAD_FINAL_SKILLS:
            continue
        if len(skill) <= 2:
            continue
        cleaned.append(skill)

    return sorted(set(cleaned))


def classify_match(score: float) -> str:
    if score >= 75:
        return "Strong Match"
    if score >= 50:
        return "Moderate Match"
    return "Low Match"


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


def generate_feedback(result: dict, resume_data: dict, job_data: dict, resume_domain_conf, job_domain_conf) -> list[str]:
    feedback = []

    if result["overall_score"] >= 80:
        feedback.append("Strong overall match for this job description.")
    elif result["overall_score"] >= 60:
        feedback.append("Moderate match. Your resume aligns with several key requirements.")
    else:
        feedback.append("Low match. Your resume may need more tailoring for this role.")

    if result["matched_skills"]:
        feedback.append("Matched skills: " + ", ".join(result["matched_skills"][:10]))

    if result["related_skills"]:
        related_messages = [
            f"{resume_skill} is related to {job_skill}"
            for job_skill, resume_skill in result["related_skills"][:5]
        ]
        feedback.append("Related skills found: " + "; ".join(related_messages))

    if result["missing_skills"]:
        feedback.append(
            "Consider adding or emphasizing these missing skills: "
            + ", ".join(result["missing_skills"][:10])
        )

    if job_data["required_years"] > 0 and resume_data["experience_years"] < job_data["required_years"]:
        feedback.append(
            f"Your resume shows about {resume_data['experience_years']} year(s) of experience, while the job asks for about {job_data['required_years']}."
        )

    feedback.append(
        f"Resume/job family alignment: resume predicted as {resume_domain_conf[0]}, job predicted as {job_domain_conf[0]}."
    )

    return feedback


def print_results(
    resume_data: dict,
    job_data: dict,
    result: dict,
    resume_raw: str,
    job_raw: str,
    resume_domain_conf,
    job_domain_conf,
) -> None:
    print("\nResume Review Result")

    print("\nResume Preview:")
    print(preview_text(resume_raw))

    print("\nJob Description Preview:")
    print(preview_text(job_raw))

    print("\nExtracted from Resume")
    print(f"\nEmail: {resume_data['email'] or 'Not found'}")
    print(f"Phone: {resume_data['phone'] or 'Not found'}")
    print(f"Years of Experience Found: {resume_data['experience_years']}")
    print(f"Education Level: {resume_data['education_level']}")
    print("Skills Found:")
    if resume_data["skills"]:
        for skill in resume_data["skills"]:
            print(f"- {skill}")
    else:
        print("None")

    print("\nExtracted from Job Description")
    print(f"\nRequired Years Found: {job_data['required_years']}")
    print(f"Required Education Level: {job_data['education_level']}")
    print("Skills Found:")
    if job_data["skills"]:
        for skill in job_data["skills"]:
            print(f"- {skill}")
    else:
        print("None")

    print("\nPredicted Job Family")
    print(f"\nResume Domain: {resume_domain_conf[0]} ({round(resume_domain_conf[1] * 100, 2)}%)")
    print(f"Job Domain: {job_domain_conf[0]} ({round(job_domain_conf[1] * 100, 2)}%)")

    print(f"\nMatch Level: {classify_match(result['overall_score'])}")
    print(f"Overall Match Score: {result['overall_score']}%")
    print(f"Semantic Text Similarity: {result['text_similarity']}%")
    print(f"Bullet/Requirement Similarity: {result['bullet_similarity']}%")
    print(f"Skill Match Score: {result['skill_score']}%")
    print(f"Experience Alignment Score: {result['experience_score']}%")
    print(f"Education Alignment Score: {result['education_score']}%")
    print(f"Job-Family Alignment Score: {result['domain_score']}%")

    print_list("Matched Skills", result["matched_skills"])

    print("\nRelated Skills:")
    if result["related_skills"]:
        for job_skill, resume_skill in result["related_skills"]:
            print(f"- {resume_skill} is related to {job_skill}")
    else:
        print("None")

    print_list("Missing Skills", result["missing_skills"])

    print("\nFeedback:")
    for item in generate_feedback(result, resume_data, job_data, resume_domain_conf, job_domain_conf):
        print(f"- {item}")


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
    print("\nChoose input mode:")
    print("1. Paste resume text")
    print("2. Load resume from 'resume input' folder (supported types are .pdf, .docx, .txt)")

    choice = input("Enter 1 or 2: ").strip()

    try:
        if choice == "1":
            resume_raw = clean_resume_text(get_multiline_text("resume text"))
        elif choice == "2":
            resume_folder = PROJECT_ROOT / "resume input"
            resume_raw = choose_resume_from_folder(resume_folder)
        else:
            print("Invalid option.")
            return
    except Exception as error:
        print(f"Resume input error: {error}")
        return

    job_raw = clean_job_text(get_multiline_text("job description text"))

    if not resume_raw.strip():
        print("Resume input is empty.")
        return

    if not job_raw.strip():
        print("Job description input is empty.")
        return

    reference_folder = PROJECT_ROOT / "data" / "Job skill reference data"
    taxonomy_terms = load_taxonomy_terms(reference_folder)

    resume_clean = preprocess_text(resume_raw)
    job_clean = preprocess_text(job_raw)

    resume_data = extract_all(resume_raw, is_job=False)
    job_data = extract_all(job_raw, is_job=True)

    resume_taxonomy_skills = extract_taxonomy_matches(resume_raw, taxonomy_terms)
    job_taxonomy_skills = extract_taxonomy_matches(job_raw, taxonomy_terms)

    resume_data["skills"] = clean_final_skills(
        combine_skills(resume_data["skills"], resume_taxonomy_skills)
    )
    job_data["skills"] = clean_final_skills(
        combine_skills(job_data["skills"], job_taxonomy_skills)
    )

    data_dir = PROJECT_ROOT / "data"
    resume_domain_conf = predict_domain(resume_raw, data_dir)
    job_domain_conf = predict_domain(job_raw, data_dir)

    result = overall_match(
        resume_text=resume_clean,
        job_text=job_clean,
        resume_skills=resume_data["skills"],
        job_skills=job_data["skills"],
        resume_years=resume_data["experience_years"],
        required_years=job_data["required_years"],
        resume_education=resume_data["education_level"],
        required_education=job_data["education_level"],
        resume_bullets=resume_data["bullets"],
        job_bullets=job_data["bullets"],
        resume_domain_conf=resume_domain_conf,
        job_domain_conf=job_domain_conf,
    )

    print_results(
        resume_data=resume_data,
        job_data=job_data,
        result=result,
        resume_raw=resume_raw,
        job_raw=job_raw,
        resume_domain_conf=resume_domain_conf,
        job_domain_conf=job_domain_conf,
    )


if __name__ == "__main__":
    main()