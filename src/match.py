from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RELATED_SKILL_GROUPS = [
    {"git", "github"},
    {"sql", "ms sql", "mssql", "mysql", "postgresql", "oracle"},
    {"api", "api development", "rest api"},
    {"html", "css", "javascript"},
    {"c#", ".net", ".net core", ".net framework", "asp.net", "asp.net mvc", "blazor"},
    {"python", "java", "c++", "javascript", "typescript"},
    {"data structures", "algorithms", "software engineering"},
]


def text_similarity(resume_text, job_text):
    docs = [resume_text, job_text]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)


def are_related(skill_a, skill_b):
    a = skill_a.lower().strip()
    b = skill_b.lower().strip()

    if a == b:
        return True

    for group in RELATED_SKILL_GROUPS:
        if a in group and b in group:
            return True

    return False


def skill_match(resume_skills, job_skills):
    matched = []
    related = []
    missing = []

    resume_lower = [x.lower().strip() for x in resume_skills]
    used_resume_skills = set()

    for job_skill in job_skills:
        job_lower = job_skill.lower().strip()

        exact_found = False
        for resume_skill in resume_lower:
            if job_lower == resume_skill:
                matched.append(job_skill)
                used_resume_skills.add(resume_skill)
                exact_found = True
                break

        if exact_found:
            continue

        related_found = False
        for resume_skill in resume_lower:
            if resume_skill in used_resume_skills:
                continue

            if are_related(job_lower, resume_skill):
                related.append((job_skill, resume_skill))
                used_resume_skills.add(resume_skill)
                related_found = True
                break

        if not related_found:
            missing.append(job_skill)

    exact_points = len(matched)
    related_points = len(related) * 0.5
    total_possible = len(job_skills)

    score = 0
    if total_possible > 0:
        score = round(((exact_points + related_points) / total_possible) * 100, 2)

    return {
        "matched": matched,
        "related": related,
        "missing": missing,
        "score": score
    }


def overall_match(resume_text, job_text, resume_skills, job_skills):
    sim_score = text_similarity(resume_text, job_text)
    skill_data = skill_match(resume_skills, job_skills)

    final_score = round((sim_score * 0.6) + (skill_data["score"] * 0.4), 2)

    return {
        "text_similarity": sim_score,
        "skill_score": skill_data["score"],
        "matched_skills": skill_data["matched"],
        "related_skills": skill_data["related"],
        "missing_skills": skill_data["missing"],
        "overall_score": final_score
    }