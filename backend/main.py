"""
main.py — Minimal FastAPI backend for testing the deployed cervical-cell model in a UI.

Run locally:
    pip install fastapi uvicorn python-multipart
    pip install -r requirements.txt   # from the inference.py repo (torch, torchvision, etc.)
    uvicorn main:app --reload --port 8000

The model downloads automatically from Hugging Face on first request (cached after that).
"""

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import CervicalCellPredictor

app = FastAPI(title="Cervical Cell Classifier API (internal)")

# This service is now called server-to-server by the Next.js API route
# (app/api/predict/route.ts), not directly by the browser — so CORS is no
# longer strictly necessary (CORS only applies to browser-initiated requests).
# Kept narrow and commented in case you ever want to hit this port directly
# for testing (e.g. curl, Postman) from a browser-based tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests — NOT per-request (that would be very slow).
predictor = CervicalCellPredictor.from_pretrained("NamrahDurrani/Huggingface-deployement")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    result = predictor.predict(pil_image)
    return result