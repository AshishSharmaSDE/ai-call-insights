FROM python:3.10-slim

# Install system deps
RUN apt update && apt install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose default port for Hugging Face
ENV PORT 7860
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
