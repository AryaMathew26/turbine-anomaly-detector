FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Some Python packages (like `catboost`, `torch`) require compilation tools.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy entire project (needed for uv sync to read version from __init__.py)
COPY . .

# Install Python dependencies using uv
# pyproject.toml is the list of dependencies we want, and uv.lock is the exact
# saved version of those dependencies.
# When we run uv sync --frozen, uv checks that these two files still match.
# If we changed pyproject.toml but forgot to update uv.lock, the command fails.
# which is good because it prevents Docker or CI from installing unexpected dependency versions.
RUN uv sync --frozen --no-dev

# Set Python path and ensure venv is in PATH
# Adds /app/src to PYTHONPATH, so Python can import your src/package_name package.
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Adds the virtual environment’s bin directory to PATH.
# Python, pip, kedro, uv, etc. from .venv can be run directly, 
# so you don’t need to use uv run or full paths.
# without this you would need to run uv python src/turbin_anomaly_detector/__main__.py
# with this you can just run python src/turbin_anomaly_detector/__main__.py
ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden in docker-compose)
CMD ["python", "--version"]