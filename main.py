import os
from pathlib import Path

import pdfplumber
from docx import Document

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
    file_path = str(file_path)
    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return read_txt_file(file_path)
    if extension == ".docx":
        return read_docx_file(file_path)
    if extension == ".pdf":
        return read_pdf_file(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def generate_feedback(result: dict) -> list[str]:
    feedback = []

    overall_score = result["overall_score"]
    missing_skills = result["missing_skills"]
    matched_skills = result["matched_skills"]

    if overall_score >= 80:
        feedback.append("Strong overall match for this job description.")
    elif overall_score >= 60:
        feedback.append("Moderate match. Your resume aligns with several key requirements.")
    else:
        feedback.append("Low match. Your resume may need more tailoring for this role.")

    if matched_skills:
        feedback.append(
            "Matched skills: " + ", ".join(matched_skills)
        )

    if missing_skills:
        feedback.append(
            "Consider adding or emphasizing these missing skills: "
            + ", ".join(missing_skills)
        )
    else:
        feedback.append("No missing skills were detected from the extracted job keywords.")

    if result["text_similarity"] < 50:
        feedback.append("Try aligning your wording more closely with the job description.")

    return feedback


def print_results(resume_data: dict, job_data: dict, result: dict) -> None:
    print("\n" + "=" * 60)
    print("RESUME REVIEW RESULTS")
    print("=" * 60)

    print(f"\nResume Email: {resume_data['email']}")
    print(f"Resume Phone: {resume_data['phone']}")
    print(f"Years of Experience Found: {resume_data['experience_years']}")

    print("\nResume Skills:")
    if resume_data["skills"]:
        print(", ".join(resume_data["skills"]))
    else:
        print("No skills found")

    print("\nJob Skills:")
    if job_data["skills"]:
        print(", ".join(job_data["skills"]))
    else:
        print("No skills found")

    print("\nScores:")
    print(f"Text Similarity Score: {result['text_similarity']}%")
    print(f"Skill Match Score: {result['skill_score']}%")
    print(f"Overall Match Score: {result['overall_score']}%")

    print("\nMatched Skills:")
    if result["matched_skills"]:
        print(", ".join(result["matched_skills"]))
    else:
        print("None")

    print("\nMissing Skills:")
    if result["missing_skills"]:
        print(", ".join(result["missing_skills"]))
    else:
        print("None")

    print("\nFeedback:")
    for item in generate_feedback(result):
        print(f"- {item}")

    print("=" * 60)


def main() -> None:
    print("Resume Review Tool")
    print("Supported file types: .txt, .docx, .pdf\n")

    resume_path = input("Enter path to resume file: ").strip()
    job_path = input("Enter path to job description file: ").strip()

    if not os.path.exists(resume_path):
        print(f"Error: Resume file not found -> {resume_path}")
        return

    if not os.path.exists(job_path):
        print(f"Error: Job description file not found -> {job_path}")
        return

    try:
        resume_raw = load_text(resume_path)
        job_raw = load_text(job_path)
    except Exception as error:
        print(f"Error reading files: {error}")
        return

    if not resume_raw.strip():
        print("Error: Resume file is empty or could not be read.")
        return

    if not job_raw.strip():
        print("Error: Job description file is empty or could not be read.")
        return

    resume_clean = preprocess_text(resume_raw)
    job_clean = preprocess_text(job_raw)

    resume_data = extract_all(resume_raw)
    job_data = {
        "skills": extract_skills(job_raw)
    }

    result = overall_match(
        resume_clean,
        job_clean,
        resume_data["skills"],
        job_data["skills"]
    )

    print_results(resume_data, job_data, result)


if __name__ == "__main__":
    main()