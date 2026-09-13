# ---------- Build React frontend ----------
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci --legacy-peer-deps

COPY frontend/ ./

ENV GENERATE_SOURCEMAP=false

RUN npm run build


# ---------- Build Python backend ----------
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY main_real_yolo.py ./
COPY yolov8n.pt ./

COPY --from=frontend-builder /frontend/build ./frontend/build

RUN mkdir -p \
    /app/data/evidence \
    /app/data/annotated \
    /app/data/reports

EXPOSE 8000

CMD ["uvicorn", "main_real_yolo:app", "--host", "0.0.0.0", "--port", "8000"]