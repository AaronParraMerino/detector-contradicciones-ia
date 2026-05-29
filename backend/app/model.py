from copy import deepcopy

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
MODEL_RESULT_CACHE: dict[tuple[str, str], dict] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()


def _normalize_label(label: str) -> str:
    return str(label).lower()


def _cache_key(premise: str, hypothesis: str) -> tuple[str, str]:
    return (premise.strip().lower(), hypothesis.strip().lower())


def _format_prediction(probs: list[float]) -> dict:
    id2label = model.config.id2label
    scores = {
        _normalize_label(id2label[i]): round(probs[i], 4)
        for i in range(len(probs))
    }
    label = max(scores, key=scores.get)

    return {
        "label": label,
        "scores": scores,
        "is_contradiction": label == "contradiction",
    }


def detect_pairs(
    pairs: list[tuple[str, str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    results: list[dict | None] = [None] * len(pairs)
    pending: list[tuple[int, str, str]] = []

    for index, (premise, hypothesis) in enumerate(pairs):
        key = _cache_key(premise, hypothesis)
        cached = MODEL_RESULT_CACHE.get(key)
        if cached is not None:
            results[index] = deepcopy(cached)
        else:
            pending.append((index, premise, hypothesis))

    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            premises = [item[1] for item in chunk]
            hypotheses = [item[2] for item in chunk]

            inputs = tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            logits = model(**inputs).logits
            probs_batch = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for (original_index, premise, hypothesis), probs in zip(chunk, probs_batch):
                prediction = _format_prediction(probs)
                MODEL_RESULT_CACHE[_cache_key(premise, hypothesis)] = deepcopy(prediction)
                results[original_index] = prediction

    return [deepcopy(result) for result in results if result is not None]


def detect_contradiction(premise: str, hypothesis: str) -> dict:
    return detect_pairs([(premise, hypothesis)])[0]
