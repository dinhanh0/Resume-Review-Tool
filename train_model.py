import re
import html
import time
from pathlib import Path

import joblib
import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "Resume data(detailed)" / "data" / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resume_category_model.joblib"
ENCODER_PATH = MODEL_DIR / "resume_label_encoder.joblib"
CACHE_PATH = PROJECT_ROOT / "data" / "detailed_resume_cache.csv"

EXPECTED_CATEGORY_NAMES = {
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"
}

MAX_FILES_PER_CATEGORY = 150
PROGRESS_EVERY = 10
USE_CACHE = True


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


def clean_text(text: str) -> str:
    text = strip_html_tags(text)
    text = clean_placeholder_text(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_pdf_file(file_path: Path) -> str:
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    return "\n".join(pages)


def collect_dataset(dataset_root: Path) -> pd.DataFrame:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Detailed resume dataset folder not found: {dataset_root}")

    records = []

    category_dirs = [
        p for p in sorted(dataset_root.iterdir())
        if p.is_dir() and p.name in EXPECTED_CATEGORY_NAMES
    ]

    print(f"Found {len(category_dirs)} categories.")
    total_kept = 0
    start_time = time.time()

    for category_index, category_dir in enumerate(category_dirs, start=1):
        category = category_dir.name
        pdf_files = sorted(category_dir.rglob("*.pdf"))

        if MAX_FILES_PER_CATEGORY is not None:
            pdf_files = pdf_files[:MAX_FILES_PER_CATEGORY]

        print(
            f"\n[{category_index}/{len(category_dirs)}] Loading category: {category} "
            f"({len(pdf_files)} files found)"
        )

        category_kept = 0

        for file_index, file_path in enumerate(pdf_files, start=1):
            try:
                raw_text = read_pdf_file(file_path)
                cleaned_text = clean_text(raw_text)
            except Exception:
                continue

            if len(cleaned_text) < 100:
                continue

            records.append({
                "text": cleaned_text,
                "label": category,
                "file_name": file_path.name
            })
            category_kept += 1
            total_kept += 1

            if file_index % PROGRESS_EVERY == 0 or file_index == len(pdf_files):
                elapsed = time.time() - start_time
                print(
                    f"  Processed {file_index}/{len(pdf_files)} | "
                    f"kept {category_kept} in {category} | "
                    f"total kept {total_kept} | elapsed {elapsed:.1f}s"
                )

    if not records:
        raise ValueError("No usable PDF resumes were found in the detailed dataset.")

    return pd.DataFrame(records)


def load_or_build_dataset(dataset_root: Path) -> pd.DataFrame:
    if USE_CACHE and CACHE_PATH.exists():
        print(f"Loading cached dataset from: {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded {len(df)} cached samples.")
        return df

    print("Building dataset from PDFs...")
    df = collect_dataset(dataset_root)

    print(f"\nSaving cache to: {CACHE_PATH}")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)

    return df


def main() -> None:
    print(f"Using dataset root: {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        print("Could not find dataset root.")
        print("Expected path:")
        print(DATASET_ROOT)
        return

    df = load_or_build_dataset(DATASET_ROOT)

    print(f"\nTotal usable samples: {len(df)}")
    print("\nSamples per category:")
    print(df["label"].value_counts().sort_index())

    unique_classes = sorted(df["label"].unique())
    print(f"\nDetected classes: {unique_classes}")

    if len(unique_classes) < 2:
        print("\nTraining stopped: fewer than 2 classes were found.")
        return

    X = df["text"]
    y_text = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=50,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                verbose=True
            )
        )
    ])

    print("\nTraining neural network model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print(classification_report(y_test_labels, y_pred_labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"Saved trained model to: {MODEL_PATH}")
    print(f"Saved label encoder to: {ENCODER_PATH}")


if __name__ == "__main__":
    main()