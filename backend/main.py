from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import re

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

# 5. Create the API Endpoint
@app.post("/predict")
async def predict_contradiction(req: PredictionRequest):
    try:
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
            "confidence": round(probability if is_contradiction else 1 - probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))