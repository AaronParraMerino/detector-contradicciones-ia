import json
import random
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
Q_FILE = BASE_DIR / "q_table.json"
STATS_FILE = BASE_DIR / "agent_stats.json"

ALPHA = 0.18
GAMMA = 0.85
EPSILON = 0.15
DEFAULT_ACTIONS = [0.0, 0.0]


def load_q():
    if Q_FILE.exists():
        with open(Q_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"LOW": DEFAULT_ACTIONS[:], "MEDIUM": DEFAULT_ACTIONS[:], "HIGH": DEFAULT_ACTIONS[:]}


def save_q(Q):
    with open(Q_FILE, "w", encoding="utf-8") as f:
        json.dump(Q, f, indent=2)


def load_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "decisions": 0,
        "updates": 0,
        "positive_rewards": 0,
        "negative_rewards": 0,
        "last_update": None,
    }


def save_stats(stats):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.5:
        return "LOW"
    if confidence < 0.8:
        return "MEDIUM"
    return "HIGH"


def get_state(confidence: float, context: dict | None = None) -> str:
    context = context or {}
    confidence_bucket = _confidence_bucket(confidence)
    paragraph_scope = context.get("paragraph_scope", "unknown_paragraph")
    distance_bucket = context.get("distance_bucket", "unknown_distance")
    topic_scope = context.get("topic_scope", "unknown_topic")
    source = context.get("source", "model")

    return "|".join(
        [
            confidence_bucket,
            paragraph_scope,
            distance_bucket,
            topic_scope,
            source,
        ]
    )


def _ensure_state(Q, state: str):
    if state not in Q:
        Q[state] = DEFAULT_ACTIONS[:]


def decide(confidence: float, context: dict | None = None) -> dict:
    Q = load_q()
    state = get_state(confidence, context)
    _ensure_state(Q, state)

    stats = load_stats()
    stats["decisions"] += 1
    save_stats(stats)

    if random.random() < EPSILON:
        action = random.randint(0, 1)
        decision_type = "exploration"
    else:
        action = int(Q[state][1] >= Q[state][0])
        decision_type = "exploitation"

    save_q(Q)
    return {
        "state": state,
        "action": action,
        "show_alert": bool(action),
        "decision_type": decision_type,
        "epsilon": EPSILON,
        "q_values": Q[state],
    }


def update(confidence: float, action: int, reward: int, context: dict | None = None):
    Q = load_q()
    state = get_state(confidence, context)
    _ensure_state(Q, state)

    best_next = max(Q[state])
    Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
    save_q(Q)

    stats = load_stats()
    stats["updates"] += 1
    if reward > 0:
        stats["positive_rewards"] += 1
    else:
        stats["negative_rewards"] += 1
    stats["last_update"] = datetime.now(timezone.utc).isoformat()
    save_stats(stats)

    return Q


def policy_stats() -> dict:
    Q = load_q()
    stats = load_stats()
    learned_states = len(Q)
    alert_states = sum(1 for values in Q.values() if values[1] >= values[0])

    return {
        **stats,
        "learned_states": learned_states,
        "alert_states": alert_states,
        "observe_states": learned_states - alert_states,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "gamma": GAMMA,
    }
