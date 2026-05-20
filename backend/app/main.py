from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import detect_contradiction
from rl_agent import decide, update
from cache import check_cache, save_correction, count_corrections

class FeedbackRequest(BaseModel):
    confidence: float
    action: int
    accepted: bool  # True = user confirmed, False = user dismissed
    correct_label: Optional[str] = None
    premise: Optional[str] = None
    hypothesis: Optional[str] = None

class ContradictionRequest(BaseModel):
    premise: str
    hypothesis: str



app = FastAPI(title="Detector de Contradicciones IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def inicio():
    return {"mensaje": "Backend funcionando correctamente"}

@app.post("/detect")
def detect(req: ContradictionRequest):
    # ── Layer 1: check override cache ──────────────────────────────────
    cached_label = check_cache(req.premise, req.hypothesis)
    if cached_label:
        return {
            "label": cached_label,
            "scores": {},
            "is_contradiction": cached_label == "contradiction",
            "source": "cache"   # tells your frontend this came from a human correction
        }

    # ── Layer 2: run the AI model normally ─────────────────────────────
    result = detect_contradiction(req.premise, req.hypothesis)
    result["source"] = "model"

    if result["is_contradiction"]:
        confidence = result["scores"]["contradiction"]
        result["rl_decision"] = decide(confidence)

    return result

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    reward = 1 if req.accepted else -1
    updated_q = update(req.confidence, req.action, reward)

    correction_saved = False
    if not req.accepted and req.correct_label:
        save_correction(req.premise, req.hypothesis, req.correct_label)
        correction_saved = True

    total = count_corrections()

    return {
        "reward": reward,
        "updated_q_table": updated_q,
        "correction_saved": correction_saved,
        "total_corrections": total,
        "ready_to_retrain": total > 0 and total % 50 == 0  # fires at 50, 100, 150...
    }
