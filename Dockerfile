FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY data/ ./data/
COPY scripts/ ./scripts/
COPY alembic/ ./alembic/
COPY alembic.ini .

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]