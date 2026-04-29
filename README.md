# Resume Review Tool

A lightweight Python prototype for analyzing resumes against job descriptions.
This project compares resume content with job requirements, extracts useful information such as skills and contact details, and returns a match score with feedback.

---

# Current Features

* Resume text preprocessing
* Skill extraction
* Email and phone detection
* Experience keyword detection
* Resume vs job description comparison
* Similarity scoring
* Missing skill detection
* Overall match percentage

---

# Project Structure

```txt
RESUME-REVIEW-TOOL/
│── src/
│   ├── preprocess.py
│   ├── extract.py
│   ├── match.py
│   ├── feedback.py
│
│── main.py
│── requirements.txt
│── README.md
```

---

# File Descriptions

## main.py

The main entry point of the program.

This file connects all modules together.

Typical workflow:

1. Load resume text
2. Load job description text
3. Preprocess both texts
4. Extract important information
5. Compare resume against job description
6. Print final score and feedback

Run with:

```bash
python main.py
```

---

## src/preprocess.py

Handles text cleaning before analysis.

Functions may include:

* converting text to lowercase
* removing punctuation
* removing stopwords
* tokenizing words
* normalizing text

Purpose:

Makes resume and job description text easier to compare accurately.

---

## src/extract.py

Extracts important information from text.

Current extraction includes:

* technical skills
* email address
* phone number
* named entities using spaCy
* years of experience (basic pattern matching)

Purpose:

Turns raw resume text into structured data.

Example:

```python
{
  "skills": ["python", "sql", "aws"],
  "email": "user@email.com"
}
```

---

## src/match.py

Compares resume data with job description data.

Current methods include:

* TF-IDF cosine similarity
* matched skills
* missing skills
* skill match percentage
* combined overall score

Purpose:

Measures how well a resume fits a job posting.

Example output:

```python
{
  "overall_score": 81.5,
  "matched_skills": ["python", "sql"],
  "missing_skills": ["docker"]
}
```

---

## src/feedback.py

Converts scoring results into human-readable suggestions.

Examples:

* Your resume strongly matches this role.
* Add Docker or Kubernetes experience.
* Improve keyword alignment with the job posting.

Purpose:

Makes technical scores useful to the user.

---

## requirements.txt

Contains all required Python libraries.

Install with:

```bash
pip install -r requirements.txt
```

---

# Setup Instructions

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Install spaCy language model

```bash
python -m spacy download en_core_web_sm
```

## 3. Run project

```bash
python main.py
```

---

# Author

Anh Dinh
Mathew Singzon

---

Built as a resume analysis and job matching prototype project using Python.
