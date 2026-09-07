# BYOC inference image for a compileml decision artifact.
# Same shape as the ch04-* images: slim Python, FastAPI/uvicorn on 8080, `serve` entrypoint.
# The compileml runtime is stdlib-only, so the package is installed with --no-deps and the
# image carries no numpy / scikit-learn / catboost.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /opt/program
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps compileml==0.4.3

COPY serve.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
