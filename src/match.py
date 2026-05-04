from domain_synonyms import ALL_RELATED_GROUPS
from semantic import semantic_similarity, best_pair_similarity, are_semantically_related

EDUCATION_ORDER = {
    "unknown": 0,
    "high_school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}

BLOCKED_RELATED_PAIRS = {
    ("clients", "presentations"),
    ("presentations", "clients"),
    ("java", "finance"),
    ("java", "accounting"),
    ("java", "fp&a"),
    ("tools", "python"),
    ("models", "python"),
}


def are_related(skill_a, skill_b):
    a = skill_a.lower().strip()
    b = skill_b.lower().strip()

    if a == b:
        return True

    if (a, b) in BLOCKED_RELATED_PAIRS:
        return False

    for group in ALL_RELATED_GROUPS:
        if a in group and b in group:
            return True

    return are_semantically_related(a, b, threshold=0.78)


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

    score = 0.0
    if total_possible > 0:
        score = round(((exact_points + related_points) / total_possible) * 100, 2)

    return {
        "matched": matched,
        "related": related,
        "missing": missing,
        "score": score
    }


def experience_alignment(resume_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 100.0
    if resume_years <= 0:
        return 0.0

    ratio = min(resume_years / required_years, 1.0)
    return round(ratio * 100, 2)


def education_alignment(resume_level: str, required_level: str) -> float:
    resume_rank = EDUCATION_ORDER.get(resume_level, 0)
    required_rank = EDUCATION_ORDER.get(required_level, 0)

    if required_rank == 0:
        return 100.0
    if resume_rank >= required_rank:
        return 100.0

    ratio = resume_rank / required_rank
    return round(ratio * 100, 2)


def domain_alignment(resume_domain_conf, job_domain_conf):
    resume_domain, _ = resume_domain_conf
    job_domain, _ = job_domain_conf

    if resume_domain == "unknown" or job_domain == "unknown":
        return 50.0

    if resume_domain == job_domain:
        return 100.0

    return 25.0


def overall_match(
    resume_text,
    job_text,
    resume_skills,
    job_skills,
    resume_years,
    required_years,
    resume_education,
    required_education,
    resume_bullets,
    job_bullets,
    resume_domain_conf,
    job_domain_conf,
):
    text_score = semantic_similarity(resume_text, job_text)
    bullet_score = best_pair_similarity(resume_bullets, job_bullets)
    skill_data = skill_match(resume_skills, job_skills)
    exp_score = experience_alignment(resume_years, required_years)
    edu_score = education_alignment(resume_education, required_education)
    domain_score = domain_alignment(resume_domain_conf, job_domain_conf)

    final_score = round(
        (skill_data["score"] * 0.30) +
        (text_score * 0.20) +
        (bullet_score * 0.20) +
        (exp_score * 0.15) +
        (edu_score * 0.10) +
        (domain_score * 0.05),
        2
    )

    return {
        "text_similarity": text_score,
        "bullet_similarity": bullet_score,
        "skill_score": skill_data["score"],
        "experience_score": exp_score,
        "education_score": edu_score,
        "domain_score": domain_score,
        "matched_skills": skill_data["matched"],
        "related_skills": skill_data["related"],
        "missing_skills": skill_data["missing"],
        "overall_score": final_score,
    }