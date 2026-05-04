from functools import lru_cache
from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    model = get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(max(0.0, score) * 100, 2)


def best_pair_similarity(items_a, items_b) -> float:
    if not items_a or not items_b:
        return 0.0

    model = get_model()
    emb_a = model.encode(items_a, convert_to_tensor=True)
    emb_b = model.encode(items_b, convert_to_tensor=True)

    sim_matrix = util.cos_sim(emb_a, emb_b)
    best_scores = sim_matrix.max(dim=1).values
    return round(float(best_scores.mean().item()) * 100, 2)


def are_semantically_related(term_a: str, term_b: str, threshold: float = 0.62) -> bool:
    model = get_model()
    embeddings = model.encode([term_a, term_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score >= threshold