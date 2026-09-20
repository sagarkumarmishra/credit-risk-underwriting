# Serving image for the scoring API.
#
# The model artefact is not baked in. It is built from 1.56 GB of source data
# that has no business inside an image, and a model that ships inside its own
# container cannot be retrained without a rebuild. Mount it at run time:
#
#   docker build -t credit-risk .
#   docker run -p 8000:8000 -v "$(pwd)/data/models:/app/data/models:ro" credit-risk
#
# /health returns 503 until a model is mounted, which is the honest answer.

FROM python:3.12-slim

# libgomp1 is LightGBM's OpenMP runtime. Without it the import fails at start
# up with a missing shared object, which is a confusing way to find out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the serving dependencies. streamlit, mlflow, matplotlib and reportlab are
# all development-time tools and would roughly triple the image for nothing.
RUN pip install --no-cache-dir \
    "pandas==2.3.3" \
    "numpy==2.5.1" \
    "scikit-learn==1.9.1" \
    "lightgbm==4.7.0" \
    "scipy==1.18.1" \
    "fastapi==0.141.1" \
    "uvicorn==0.53.0" \
    "pydantic==2.13.5"

COPY src/ /app/src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/data/models/model.pkl

# Run as a non-root user. Nothing here needs to write to the filesystem.
RUN useradd --create-home --shell /bin/false scorer \
    && chown -R scorer:scorer /app
USER scorer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
