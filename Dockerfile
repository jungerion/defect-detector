FROM python:3.12-slim

# Install uv (fast dependency installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# RUN uv pip install --system .
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision --no-cache-dir
RUN uv pip install --system --no-cache-dir .

COPY config ./config
COPY models ./models

ENV PROJECT_ROOT=/app

EXPOSE 8000

CMD ["uvicorn", "defect_detector.api.main:app", "--host", "0.0.0.0", "--port", "8000"]