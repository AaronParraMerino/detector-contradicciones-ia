import json
import os

Q_FILE = "q_table.json"

# Hyperparameters
ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
EPSILON = 0.2 # exploration rate (20% random, 80% use learned policy)

def load_q():
    if os.path.exists(Q_FILE):
        with open(Q_FILE) as f:
            return json.load(f)
    return {"LOW": [0.0, 0.0], "MEDIUM": [0.0, 0.0], "HIGH": [0.0, 0.0]}

def save_q(Q):
    with open(Q_FILE, "w") as f:
        json.dump(Q, f)

def get_state(confidence: float) -> str:
    if confidence < 0.5:
        return "LOW"
    elif confidence < 0.8:
        return "MEDIUM"
    else:
        return "HIGH"

def decide(confidence: float) -> dict:
    """Agent decides whether to show the alert."""
    import random
    Q = load_q()
    state = get_state(confidence)

    # Epsilon-greedy: explore randomly sometimes
    if random.random() < EPSILON:
        action = random.randint(0, 1)
        decision_type = "exploration"
    else:
        action = int(Q[state][1] >= Q[state][0])  # pick best known action
        decision_type = "exploitation"

    return {
        "state": state,
        "action": action,        # 0 = no alert, 1 = show alert
        "show_alert": bool(action),
        "decision_type": decision_type
    }

def update(confidence: float, action: int, reward: int):
    """User reacted — update Q-table."""
    Q = load_q()
    state = get_state(confidence)

    # Q-learning formula
    best_next = max(Q[state])
    Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

    save_q(Q)
    return Q