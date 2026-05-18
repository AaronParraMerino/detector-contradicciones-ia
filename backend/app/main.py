from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Detector de Contradicciones IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def inicio():
    return {"mensaje": "Backend funcionando correctamente"}

@app.post("/analizar")
def analizar_texto(data: dict):
    texto = data.get("texto", "")

    return {
        "texto_recibido": texto,
        "contradicciones": [],
        "mensaje": "Análisis pendiente de integrar con el modelo PLN"
    }

@app.post("/feedback")
def guardar_feedback(data: dict):
    return {
        "mensaje": "Retroalimentación recibida",
        "data": data
    }