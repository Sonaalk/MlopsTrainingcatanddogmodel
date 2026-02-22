from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import time
import logging

from src.inference.model import load_model, predict_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

request_count = 0
app = FastAPI()

templates = Jinja2Templates(directory="src/inference/templates")
model = load_model()

@app.get("/health")
def health():
    logging.info("Health check called")
    return {"status": "UP"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    start_time = time.time()
    request_count += 1

    image = Image.open(file.file).convert("RGB")
    result = predict_image(model, image)

    latency = time.time() - start_time

    logging.info(
        f"request_id={request_count} "
        f"endpoint=/predict "
        f"latency={latency:.4f}s"
    )

    return {
        "prediction": result,
        "latency_seconds": latency,
        "request_count": request_count
    }