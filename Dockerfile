FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code and model
COPY src ./src


EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]