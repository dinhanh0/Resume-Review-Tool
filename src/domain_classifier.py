from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from domain_synonyms import DOMAIN_GROUPS
from semantic import semantic_similarity


MODEL_PATH = Path("models/domain_classifier.joblib")


def classify_by_prototypes(text: str):
    best_domain = "unknown"
    best_score = 0.0

    for domain_name, domain_data in DOMAIN_GROUPS.items():
        score = semantic_similarity(text, domain_data["prototype"])
        if score > best_score:
            best_score = score
            best_domain = domain_name

    confidence = round(best_score / 100.0, 4)
    return best_domain, confidence


def train_domain_classifier(data_dir: Path):
    resume_path = data_dir / "Resume data" / "Resume.csv"

    if not resume_path.exists():
        return None

    try:
        df = pd.read_csv(resume_path)
    except Exception:
        return None

    if "Resume_html" not in df.columns or "Category" not in df.columns:
        return None

    df = df.dropna(subset=["Resume_html", "Category"]).copy()
    if len(df) < 20:
        return None

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(df["Resume_html"].astype(str), df["Category"].astype(str))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train_classifier(data_dir: Path):
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    return train_domain_classifier(data_dir)


def predict_domain(text: str, data_dir: Path):
    model = load_or_train_classifier(data_dir)

    if model is None:
        return classify_by_prototypes(text)

    try:
        probs = model.predict_proba([text])[0]
        labels = model.classes_
        max_idx = probs.argmax()
        label = str(labels[max_idx])
        confidence = float(probs[max_idx])

        if confidence < 0.35:
            return classify_by_prototypes(text)

        return label, round(confidence, 4)
    except Exception:
        return classify_by_prototypes(text)