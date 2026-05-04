import re
import html
from datetime import datetime

import spacy

from domain_synonyms import DOMAIN_GROUPS

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


COMMON_SKILLS = sorted({
    # software / tech
    "python", "java", "javascript", "typescript", "c++", "c#", ".net", ".net core",
    ".net framework", "asp.net", "asp.net mvc", "blazor", "react", "node.js", "node",
    "html", "css", "sql", "ms sql", "mssql", "mysql", "postgresql", "oracle", "api",
    "api development", "rest api", "git", "github", "ci/cd", "ci cd", "docker",
    "kubernetes", "linux", "aws", "azure", "excel", "microsoft office", "word",
    "powerpoint", "pandas", "machine learning", "data analysis", "data structures",
    "algorithms", "software engineering", "full stack", "full-stack", "database management",
    "web development", "software development",

    # general transferable
    "customer service", "communication", "teamwork", "problem solving", "training",
    "multitasking", "time management", "attention to detail", "organization", "adaptability",

    # restaurant / hospitality / service
    "hospitality", "front office", "server", "serving", "food service", "guest service",
    "food safety", "sanitation", "order taking", "cash handling", "point of sale", "pos",
    "menu knowledge", "food handling", "beverage service", "restaurant operations", "cleaning",
    "serving tables",

    # retail / sales
    "sales", "retail", "inventory", "merchandising", "upselling", "transactions",
    "stocking", "cash register", "store operations",

    # healthcare support
    "healthcare", "patient care", "resident care", "medical records", "caregiving",
    "health and safety",

    # finance / office / business
    "finance", "accounting", "reporting", "budgeting", "forecasting", "business analysis",
    "administration", "office administration", "data entry", "accounts payable",
    "accounts receivable", "general ledger", "month-end close", "cpa", "investor reporting",
    "bank reconciliations", "real estate accounting", "fp&a", "financial planning",
    "financial analysis", "corporate finance", "variance analysis", "financial modeling",
    "financial models", "analytics", "data-driven insights", "business acumen",
    "dashboards", "kpis", "board decks", "executive presentations", "scenario planning",
    "strategic planning", "resource allocation", "p&l", "p&l ownership", "sku",
    "portfolio management", "brand portfolio", "cpg", "consumer packaged goods",
    "pricing", "promotion", "promotional roi", "trade spend", "gross-to-net",
    "category management", "supply chain", "supply chain costing", "finance executive team",

    # logistics / operations
    "logistics", "supply chain", "operations", "scheduling", "coordination", "shipping",
    "receiving", "warehouse", "delivery", "inventory management",

    # HR / people / business partner
    "human resources", "hr", "hr business partner", "talent management", "workforce planning",
    "organizational development", "change management", "stakeholder engagement", "compliance",
    "labor relations", "employee relations", "coaching", "career development",
    "workforce analytics", "policy interpretation", "presentations", "data-driven insights",
    "strategic alignment", "organizational assessments", "employee experience", "retention",
    "leadership coaching", "training needs", "organizational capacity", "financial acumen",
    "business acumen", "analytics", "dashboards", "change leadership",
    "stakeholder management", "communication strategy", "hr policy", "performance management",
    "talent development", "culture initiatives", "business partner", "organizational strategy",

    # engineering / environmental
    "engineering", "civil engineering", "environmental engineering", "chemical engineering",
    "mechanical engineering", "environmental assessments", "remediation", "permitting",
    "environmental regulations", "regulatory agencies", "project manager", "technical lead",
    "technical oversight", "proposals", "negotiations", "clients", "stakeholders",
    "staff management", "professional development", "environmental engineers",
    "california professional engineer", "rcra", "cercla", "construction management",
    "budget", "quality objectives", "transportation",

    # construction / labor
    "construction", "construction labor", "laborer", "physical labor", "field work",
    "safety", "travel", "manufacturing", "welding", "materials", "tools",
    "debris removal", "site cleanup", "construction debris", "sandbags",
    "insulating material", "geosynthetic", "grinding", "sanding", "polishing",
    "safe work environment",

    # forensic / lab science
    "forensic science", "forensic analysis", "forensic scientist", "crime lab",
    "latent print", "latent print examination", "criminalistics", "laboratory",
    "evidence examination", "evidence analysis", "biology", "chemistry",
    "natural sciences", "physical sciences", "case documentation", "quality audits",
    "courtroom testimony", "technical review", "administrative review",
    "troubleshooting", "calibration", "validation", "proficiency testing",
    "science-related coursework", "anab", "iso 17025", "osac", "nist",
    "fingerprint classification", "comparison", "analysis", "monitoring"
})


VAGUE_SKILLS = {
    "science", "design", "leadership", "database", "databases",
    "marketing", "project management", "fast-paced", "team-oriented",
    "rapport", "attentive service", "level", "levels", "writing",
    "environment", "construction laborers", "mechanical", "civil", "chemical",
    "administrative", "materials", "models", "tools", "delivery"
}


MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
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
    text = text.replace("’", "'")
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

    def should_skip_skill(skill_lower: str, text_lower: str) -> bool:
        if skill_lower == "java":
            if re.search(r"\bjava house\b", text_lower):
                if not re.search(r"\bjava\b(?!\s*house)", text_lower):
                    return True
        return False

    for skill in COMMON_SKILLS:
        skill_lower = skill.lower()

        if should_skip_skill(skill_lower, text_lower):
            continue

        if skill_lower in special_skills:
            if skill_lower in text_lower:
                found.add(skill_lower)
            continue

        pattern = rf"\b{re.escape(skill_lower)}\b"
        if re.search(pattern, text_lower):
            found.add(skill_lower)

    for domain_data in DOMAIN_GROUPS.values():
        for skill in domain_data["skills"]:
            skill_lower = skill.lower()

            if should_skip_skill(skill_lower, text_lower):
                continue

            pattern = rf"\b{re.escape(skill_lower)}\b"
            if re.search(pattern, text_lower):
                found.add(skill_lower)

    found = {skill for skill in found if skill not in VAGUE_SKILLS}
    found = {skill for skill in found if len(skill) > 2}
    return sorted(found)

def extract_email(text: str):
    text = normalize_text(text)
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_phone(text: str):
    text = normalize_text(text)
    match = re.search(r"(\+?\d[\d\-\(\) ]{8,}\d)", text)
    return match.group(0) if match else None


def detect_education_level(text: str) -> str:
    text_lower = normalize_text(text).lower()

    found_levels = []

    if "doctor of philosophy" in text_lower or re.search(r"\bphd\b", text_lower):
        found_levels.append("phd")
    if re.search(r"\bmaster(?:'s|s)?\b", text_lower) or "master of" in text_lower:
        found_levels.append("master")
    if re.search(r"\bbachelor(?:'s|s)?\b", text_lower) or "bachelor of" in text_lower:
        found_levels.append("bachelor")
    if re.search(r"\bassociate\b", text_lower) or "associate of" in text_lower:
        found_levels.append("associate")
    if "high school" in text_lower:
        found_levels.append("high_school")

    order = {
        "high_school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5,
    }

    if not found_levels:
        return "unknown"

    return max(found_levels, key=lambda level: order[level])


def extract_required_education_level(text: str) -> str:
    text_lower = normalize_text(text).lower()

    section_patterns = [
        r"(minimum education required.*?)(?:minimum experience required|special requirements|preferred requirements|additional information|primary location|job function|education:|$)",
        r"(education:.*?)(?:experience:|technical|leadership|qualifications|$)",
        r"(forensic scientist iii requirements:.*?)(?:special requirements|forensic scientist ii requirements:|$)",
        r"(forensic scientist ii requirements:.*?)(?:special requirements|forensic scientist iii:|$)",
        r"(minimum requirements.*?)(?:preferred requirements|additional information|primary location|job function|$)",
        r"(qualifications.*?)(?:additional information|about aecom|what makes aecom|$)",
    ]

    education_text = text_lower
    for pattern in section_patterns:
        match = re.search(pattern, text_lower, flags=re.IGNORECASE | re.DOTALL)
        if match:
            education_text = match.group(1)
            break

    degree_sentence_match = re.search(
        r"([^.•\n]*?(bachelor(?:'s|s)?|bachelor of|\bbs\b|\bba\b|associate|high school|master(?:'s|s)?|master of)[^.•\n]*)",
        education_text,
        flags=re.IGNORECASE,
    )

    if degree_sentence_match:
        education_text = degree_sentence_match.group(1)

    found_levels = []

    if "doctor of philosophy" in education_text or re.search(r"\bphd\b", education_text):
        found_levels.append("phd")
    if re.search(r"\bmaster(?:'s|s)?\b", education_text) or "master of" in education_text:
        found_levels.append("master")
    if re.search(r"\bbachelor(?:'s|s)?\b", education_text) or "bachelor of" in education_text:
        found_levels.append("bachelor")
    if re.search(r"\bbs\b", education_text):
        found_levels.append("bachelor")
    if re.search(r"\bba\b", education_text):
        found_levels.append("bachelor")
    if re.search(r"\bassociate\b", education_text) or "associate of" in education_text:
        found_levels.append("associate")
    if "high school" in education_text:
        found_levels.append("high_school")

    order = {
        "high_school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5,
    }

    if not found_levels:
        return "unknown"

    return max(found_levels, key=lambda level: order[level])


def extract_required_years(text: str) -> float:
    text_lower = normalize_text(text).lower()

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    def extract_values(target_text: str):
        values = []

        digit_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*years?", target_text)
        values.extend(float(x) for x in digit_matches)

        paren_digit_matches = re.findall(r"\((\d+(?:\.\d+)?)\)\+?\s*years?", target_text)
        values.extend(float(x) for x in paren_digit_matches)

        word_digit_matches = re.findall(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*\((\d+(?:\.\d+)?)\)\+?\s*years?",
            target_text
        )
        for _, digit in word_digit_matches:
            values.append(float(digit))

        word_only_matches = re.findall(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\+?\s*years?",
            target_text
        )
        for word in word_only_matches:
            values.append(float(number_words[word]))

        return values

    section_patterns = [
        r"(minimum experience required.*?)(?:special requirements|preferred requirements|additional information|primary location|job function|$)",
        r"(experience:.*?)(?:technical|leadership|education:|$)",
        r"(forensic scientist iii requirements:.*?)(?:special requirements|forensic scientist ii requirements:|$)",
        r"(forensic scientist ii requirements:.*?)(?:special requirements|forensic scientist iii:|$)",
        r"(minimum requirements.*?)(?:preferred requirements|additional information|primary location|job function|$)",
        r"(qualifications.*?)(?:preferred requirements|additional information|about aecom|what makes aecom|$)",
    ]

    section_text = None
    for pattern in section_patterns:
        match = re.search(pattern, text_lower, flags=re.IGNORECASE | re.DOTALL)
        if match:
            section_text = match.group(1)
            break

    target_text = section_text if section_text else text_lower
    target_text = re.split(r"preferred requirements", target_text, maxsplit=1)[0]
    values = extract_values(target_text)

    if not values:
        return 0.0

    if "with a degree" in target_text and "without a degree" in target_text:
        return min(values)

    return max(values)


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
    explicit_years = extract_required_years(text)
    ranged_years = extract_date_range_experience(text)

    best_years = max(explicit_years, ranged_years)

    if best_years == 0:
        return 0
    if float(best_years).is_integer():
        return int(best_years)

    return best_years


def split_resume_bullets(text: str):
    clean_text = normalize_text(text)
    parts = re.split(r"[•\n\-]+", clean_text)
    bullets = [part.strip() for part in parts if len(part.strip()) > 20]
    return bullets[:30]


def split_job_requirements(text: str):
    clean_text = normalize_text(text)
    parts = re.split(r"[•\n;]+", clean_text)
    reqs = [part.strip() for part in parts if len(part.strip()) > 20]
    return reqs[:30]


def extract_entities(text: str):
    entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}

    if nlp is None:
        return entities

    clean_text = normalize_text(text)
    doc = nlp(clean_text)

    for ent in doc.ents:
        if ent.label_ in entities:
            value = ent.text.strip()
            if 2 <= len(value) <= 60:
                entities[ent.label_].append(value)

    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities


def extract_all(text: str, is_job: bool = False):
    clean_text = normalize_text(text)

    return {
        "email": extract_email(clean_text),
        "phone": extract_phone(clean_text),
        "skills": extract_skills(clean_text),
        "experience_years": extract_years_experience(clean_text) if not is_job else 0,
        "required_years": extract_required_years(clean_text) if is_job else 0,
        "education_level": extract_required_education_level(clean_text) if is_job else detect_education_level(clean_text),
        "bullets": split_job_requirements(clean_text) if is_job else split_resume_bullets(clean_text),
        "entities": extract_entities(clean_text),
    }