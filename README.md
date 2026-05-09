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
Main program entry point.

It:
1. gets resume input
2. gets job description input
3. preprocesses text
4. extracts features
5. compares both documents
6. prints score and feedback

Run with:

```bash
python main.py
```

## train_model.py

Trains the resume domain-classification model.

It:

loads labeled resume PDFs
cleans and caches text
trains the TF-IDF + MLP model
saves the model and label encoder

Run with:

python train_model.py

## src/preprocess.py

Cleans and normalizes text before analysis.

Examples:

lowercase conversion
punctuation cleanup
tokenization
stopword removal

## src/extract.py

Extracts structured information from text.

Current extraction includes:

skills
email
phone number
education level
years of experience
named entities


## src/feedback.py

Generates user-friendly feedback from the matching results.

Examples:

matched skills
missing skills
suggestions for improvement
overall match comments


## src/match.py

Computes similarity and final scoring.

Includes:

skill matching
semantic similarity
experience alignment
education alignment
domain alignment
overall weighted score


## src/semantic.py

Handles semantic similarity.

Used to compare resume and job description meaning with Sentence-BERT embeddings.


## src/taxonomy.py

Loads reference skill data and matches taxonomy terms found in text.

Used to improve skill detection.


## src/domain_classifier.py

Loads the saved classifier model and predicts the domain/job family of input text.

Examples:

finance
HR
healthcare
software

## src/domain_synonyms.py

Stores domain groups, related skill groups, and synonym-style mappings used in matching and classification support.

## requirements.txt

Lists Python packages needed to run the project.

Install with:

pip install -r requirements.txt
models/

Stores saved trained model files.

Main files:

resume_category_model.joblib
resume_label_encoder.joblib
data/

Contains project datasets and reference data.

Examples:

resume/job sample datasets
detailed resume training data
job skill reference data
cached cleaned resume data
resume input/

Folder for resume files used by file-input mode.

Supported types:

.pdf
.docx
.txt


---

# Setup Instructions

To setup and run the project

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

To retrain the model

```bash
.\.venv\Scripts\python.exe train_model.py
```

One more practical note: this assumes Python 3.12 is installed on the computer already.

# Author

Anh Dinh
Mathew Singzon

---

Built as a resume analysis and job matching prototype project using Python.
