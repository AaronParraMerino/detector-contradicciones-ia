from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import re
import os
import pandas as pd
from collections import Counter

# 1. Define the exact same architecture
class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super(SiameseLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        linear_input_dim = (hidden_dim * 2) * 4 
        self.fc = nn.Sequential(
            nn.Linear(linear_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward_once(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

    def forward(self, premise, hypothesis):
        u = self.forward_once(premise)
        v = self.forward_once(hypothesis)
        abs_diff = torch.abs(u - v)
        mul = u * v
        combined = torch.cat((u, v, abs_diff, mul), dim=1)
        return self.fc(combined)

# 2. Initialize the app and global variables
app = FastAPI(title="Detector de contradicciones")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load resources
with open('./peso_palabras.json', 'r', encoding='utf-8') as f:
    word2idx = json.load(f)

model = SiameseLSTM(len(word2idx), 100, 128, 1).to(device)
model.load_state_dict(torch.load('contradicciones_umss_modelo.pth', map_location=device))
model.eval()

# 3. Define the Regex Tokenizer
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# 4. Define the expected JSON payload
class PredictionRequest(BaseModel):
    premise: str
    hypothesis: str

class FeedbackRequest(BaseModel):
    premise: str
    hypothesis: str
    prediction: str
    correct_label: str
    confidence: float

def save_feedback(data: dict):
    feedback_file_json = "feedback.json"
    feedback_file_parquet = "feedback.parquet"
    
    # Save to JSON
    if os.path.exists(feedback_file_json):
        with open(feedback_file_json, "r") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    else:
        records = []
    
    records.append(data)
    with open(feedback_file_json, "w") as f:
        json.dump(records, f, indent=4)
        
    # Save to Parquet
    try:
        df = pd.DataFrame(records)
        df.to_parquet(feedback_file_parquet, index=False)
    except Exception as e:
        print(f"Error saving parquet: {e}")

def check_cache(premise: str, hypothesis: str):
    feedback_file_json = "feedback.json"
    if not os.path.exists(feedback_file_json):
        return None
        
    try:
        with open(feedback_file_json, "r") as f:
            records = json.load(f)
    except Exception:
        return None
            
    p = premise.strip().lower()
    h = hypothesis.strip().lower()
    
    matches = [r for r in records if r.get("premise", "").strip().lower() == p and 
                                     r.get("hypothesis", "").strip().lower() == h]
    
    labels = [r.get("correct_label") for r in matches if r.get("correct_label")]
    if not labels:
        return None
        
    counter = Counter(labels)
    most_common_label, count = counter.most_common(1)[0]
    
    if count >= 2:
        return most_common_label
    return None

# 5. Create the API Endpoint
@app.post("/predict")
async def predict_contradiction(req: PredictionRequest):
    try:
        cached_label = check_cache(req.premise, req.hypothesis)
        if cached_label:
            return {
                "prediction": cached_label,
                "confidence": 1.0,
                "source": "cache"
            }

        premise_tokens = tokenize(req.premise)
        hypo_tokens = tokenize(req.hypothesis)
        
        premise_indices = [word2idx.get(t, 1) for t in premise_tokens]
        hypo_indices = [word2idx.get(t, 1) for t in hypo_tokens]
        
        premise_tensor = torch.tensor(premise_indices).unsqueeze(0).to(device)
        hypo_tensor = torch.tensor(hypo_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(premise_tensor, hypo_tensor)
            probability = torch.sigmoid(output).item()
            is_contradiction = probability > 0.5
            
        return {
            "prediction": "contradiction" if is_contradiction else "not_contradiction",
            "confidence": round(probability if is_contradiction else 1 - probability, 4),
            "source": "model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    try:
        data = req.dict()
        background_tasks.add_task(save_feedback, data)
        return {"message": "Feedback saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))