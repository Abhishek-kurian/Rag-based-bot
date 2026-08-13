FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/torch/sentence_transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app

RUN mkdir -p /home/user/.cache/huggingface \
    /home/user/.cache/torch/sentence_transformers \
    && chown -R user:user /home/user

USER user

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
