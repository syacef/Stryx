FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# You will need to install python3 manually now
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python if needed
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

# Install Torch Nightly for CUDA 12.8/sm_120
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install the rest of the requirements (remove torch from requirements.txt or let pip handle satisfied deps)
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY student_tinyvit.onnx labels.txt student_tinyvit.onnx.data .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV MODEL_CHECKPOINT=/app/student_tinyvit.onnx

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "main.py"]
