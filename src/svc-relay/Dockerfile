FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    vlc \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m stryxuser
USER stryxuser

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY *.py .

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/stryxuser/.local/bin:${PATH}"

EXPOSE 8000

CMD ["python", "main.py"]
