FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serve.py .
COPY model.py .

# Set environment variables
ENV NORMALIZE=true
ENV MODEL_NAME="openai/clip-vit-large-patch14"

# Pre-download the model
RUN python model.py

# Expose the application port
EXPOSE 8080

# Start FastAPI server with Uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
