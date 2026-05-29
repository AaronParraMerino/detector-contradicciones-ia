import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "corrections_cache.json"
TRAINING_FILE = BASE_DIR / "corrections.jsonl"
SIMILARITY_THRESHOLD = 0.72
STOPWORDS = {
    "al",
    "del",
    "el",
    "ella",
    "ellas",
    "ellos",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "una",
    "un",
    "uno",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", _normalize_text(text))
        if token not in STOPWORDS
    }


def _key(premise: str, hypothesis: str) -> str:
    return f"{premise.strip().lower()}|||{hypothesis.strip().lower()}"


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return 0
    union = first | second
    return len(first & second) / len(union) if union else 0


def _pair_similarity(
    premise: str,
    hypothesis: str,
    correction: dict,
) -> tuple[float, bool]:
    premise_tokens = _tokens(premise)
    hypothesis_tokens = _tokens(hypothesis)
    stored_premise_tokens = _tokens(correction.get("premise", ""))
    stored_hypothesis_tokens = _tokens(correction.get("hypothesis", ""))

    direct = (
        _jaccard(premise_tokens, stored_premise_tokens)
        + _jaccard(hypothesis_tokens, stored_hypothesis_tokens)
    ) / 2
    reversed_pair = (
        _jaccard(premise_tokens, stored_hypothesis_tokens)
        + _jaccard(hypothesis_tokens, stored_premise_tokens)
    ) / 2

    if reversed_pair > direct:
        return reversed_pair, True
    return direct, False


def check_cache(premise: str, hypothesis: str):
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get(_key(premise, hypothesis))


def load_corrections(limit: int | None = None) -> list[dict]:
    if not TRAINING_FILE.exists():
        return []

    corrections = []
    with open(TRAINING_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                corrections.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if limit is None:
        return corrections
    return corrections[-limit:]


def find_similar_correction(
    premise: str,
    hypothesis: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict | None:
    best_match = None
    best_score = 0

    for correction in load_corrections(limit=400):
        score, reversed_pair = _pair_similarity(premise, hypothesis, correction)
        if score > best_score:
            best_score = score
            best_match = {
                "label": correction.get("label"),
                "similarity": round(score, 4),
                "reversed": reversed_pair,
                "matched_premise": correction.get("premise"),
                "matched_hypothesis": correction.get("hypothesis"),
            }

    if best_match and best_match["label"] and best_score >= threshold:
        return best_match
    return None


def save_correction(premise: str, hypothesis: str, correct_label: str):
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)

    cache[_key(premise, hypothesis)] = correct_label
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    entry = {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": correct_label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "premise_tokens": sorted(_tokens(premise)),
        "hypothesis_tokens": sorted(_tokens(hypothesis)),
    }
    with open(TRAINING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_corrections() -> int:
    return len(load_corrections())


def memory_stats() -> dict:
    corrections = load_corrections()
    labels: dict[str, int] = {}
    for correction in corrections:
        label = correction.get("label", "unknown")
        labels[label] = labels.get(label, 0) + 1

    exact_items = 0
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            exact_items = len(json.load(f))

    return {
        "corrections": len(corrections),
        "exact_items": exact_items,
        "labels": labels,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
