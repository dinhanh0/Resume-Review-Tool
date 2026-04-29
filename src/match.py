# src/match.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz


def text_similarity(resume_text, job_text):
    """
    TF-IDF cosine similarity
    """
    docs = [resume_text, job_text]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)


def skill_match(resume_skills, job_skills):
    """
    Compare skill lists
    """
    matched = []
    missing = []

    resume_lower = [x.lower() for x in resume_skills]

    for skill in job_skills:
        if skill.lower() in resume_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    score = 0
    if len(job_skills) > 0:
        score = round((len(matched) / len(job_skills)) * 100, 2)

    return {
        "matched": matched,
        "missing": missing,
        "score": score
    }


def fuzzy_title_match(resume_text, target_title):
    """
    Match job title
    """
    return fuzz.partial_ratio(resume_text.lower(), target_title.lower())


def overall_match(resume_text, job_text, resume_skills, job_skills):
    """
    Combined score
    """
    sim_score = text_similarity(resume_text, job_text)
    skill_data = skill_match(resume_skills, job_skills)

    final_score = round((sim_score * 0.6) + (skill_data["score"] * 0.4), 2)

    return {
        "text_similarity": sim_score,
        "skill_score": skill_data["score"],
        "matched_skills": skill_data["matched"],
        "missing_skills": skill_data["missing"],
        "overall_score": final_score
    }