FROM python:3.11-slim

# Install system dependencies (Tesseract OCR and clean up apt cache)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (preserving monorepo structure)
COPY . /app/backend

# Configure python path to find backend modules
ENV PYTHONPATH=/app

EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
