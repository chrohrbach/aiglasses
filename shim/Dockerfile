FROM python:3.12-slim

RUN useradd --create-home --uid 1000 shimuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# Pre-create the volume mount target with the right ownership so the named
# volume (photo_cache) inherits it on first mount. Without this, /app/data
# ends up root-owned and shimuser can't create the photos subdir.
RUN mkdir -p /app/data/photos && chown -R shimuser:shimuser /app/data

USER shimuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
