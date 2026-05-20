# model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def detect_contradiction(premise: str, hypothesis: str) -> dict:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label  # e.g. {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}

    scores = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    label = max(scores, key=scores.get)
    print(label == "contradiction")

    return {
        "label": label,
        "scores": scores,
        "is_contradiction": label == "contradiction"
    }