from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from src.inference.model import load_model, predict_image

app = FastAPI(title="Cats vs Dogs Inference Service")

templates = Jinja2Templates(directory="src/inference/templates")
model = load_model()

@app.get("/health")
def health():
    return {"status": "UP"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(model, image)
    return result