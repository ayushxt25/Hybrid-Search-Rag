FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY backend/app ./backend/app
RUN python -m pip install --no-cache-dir --upgrade build \
    && python -m build --wheel --outdir /dist

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir -p /app "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app
COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl \
    && python -m pip install --no-cache-dir "sentence-transformers==5.6.0" \
    && rm -rf /tmp/*.whl

USER 10001:10001
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
