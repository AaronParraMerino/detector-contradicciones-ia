import json, os

CACHE_FILE = "corrections_cache.json"
TRAINING_FILE = "corrections.jsonl"

def _key(premise: str, hypothesis: str) -> str:
    # Normalise so "El gato  " and "El gato" match the same key
    return f"{premise.strip().lower()}|||{hypothesis.strip().lower()}"

def check_cache(premise: str, hypothesis: str):
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get(_key(premise, hypothesis))  # returns "entailment" or None

def save_correction(premise: str, hypothesis: str, correct_label: str):
    # 1. Write to instant override cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    cache[_key(premise, hypothesis)] = correct_label
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    # 2. Append to training file for future retraining
    entry = {"premise": premise, "hypothesis": hypothesis, "label": correct_label}
    with open(TRAINING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def count_corrections() -> int:
    if not os.path.exists(TRAINING_FILE):
        return 0
    with open(TRAINING_FILE, encoding="utf-8") as f:
        return sum(1 for _ in f)