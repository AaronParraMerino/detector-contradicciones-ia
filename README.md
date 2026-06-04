# Detector de Contradicciones IA

Aplicacion web para detectar posibles contradicciones entre dos fragmentos de texto usando Procesamiento de Lenguaje Natural. El frontend permite escribir un texto en formato libre; internamente lo divide en premisa e hipotesis y lo envia al backend para obtener una prediccion.

## Estado actual del proyecto

- Frontend: React + Vite + Tailwind CSS.
- Backend: Python + FastAPI + PyTorch.
- Modelo IA: red neuronal `SiameseLSTM` entrenada para clasificacion binaria.
- Clases actuales del backend: `contradiction` y `not_contradiction`.
- Feedback: el usuario puede corregir predicciones; el backend guarda retroalimentacion en `feedback.json` y `feedback.parquet`.
- Memoria simple: si una misma correccion aparece al menos 2 veces para la misma premisa e hipotesis, el backend responde desde cache.

> Nota: esta rama no usa SQLite ni implementa Q-learning/RL formal. La parte de aprendizaje actual es retroalimentacion humana persistida y cache por votacion.

## Estructura principal

```txt
backend/
  main.py                         API FastAPI y logica de inferencia
  requirements.txt                Dependencias Python
  peso_palabras.json              Vocabulario palabra -> indice
  contradicciones_umss_modelo.pth Pesos del modelo PyTorch

frontend/
  src/App.jsx                     Interfaz principal
  src/index.css                   Estilos y Tailwind
  package.json                    Scripts y dependencias frontend

contradicciones_entrenar.ipynb    Notebook de entrenamiento
INFORME_TECNICO_PROYECTO.md       Informe tecnico detallado del repositorio
```

## Ejecutar backend

Desde la raiz del proyecto:

```bash
cd backend
source .venv/Scripts/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

En PowerShell:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Documentacion de la API:

```txt
http://127.0.0.1:8000/docs
```

Importante: el backend debe ejecutarse desde la carpeta `backend`, porque carga `peso_palabras.json` y `contradicciones_umss_modelo.pth` con rutas relativas.

## Ejecutar frontend

Desde la raiz del proyecto:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

URL esperada:

```txt
http://127.0.0.1:5173
```

El frontend espera que el backend este disponible en:

```txt
http://localhost:8000
```

## Endpoints principales

### `POST /predict`

Recibe:

```json
{
  "premise": "texto base",
  "hypothesis": "texto a comparar"
}
```

Devuelve:

```json
{
  "prediction": "contradiction",
  "confidence": 0.8732,
  "source": "model"
}
```

### `POST /feedback`

Recibe correcciones del usuario:

```json
{
  "premise": "texto base",
  "hypothesis": "texto a comparar",
  "prediction": "contradiction",
  "correct_label": "not_contradiction",
  "confidence": 0.8732
}
```