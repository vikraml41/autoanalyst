# Use official Python image instead of jupyter to avoid conflicts
FROM python:3.9-slim

# Install system dependencies for scipy/sklearn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install scipy and sklearn FIRST (most likely to fail)
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir scipy==1.10.1
RUN pip install --no-cache-dir scikit-learn==1.3.0

# Install pandas
RUN pip install --no-cache-dir pandas==2.1.3

# Install FastAPI with specific anyio version to avoid conflicts
RUN pip install --no-cache-dir \
    "anyio<4.0.0,>=3.1.0" \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0"

# Install other packages
RUN pip install --no-cache-dir \
    yfinance \
    beautifulsoup4 \
    textblob \
    praw \
    python-multipart \
    aiofiles \
    requests \
    plotly \
    matplotlib \
    seaborn

# Copy application files
COPY . .

# Debug: List what was copied
RUN echo "=== Contents of /app ===" && ls -la /app/
RUN echo "=== Contents of /app/data (if exists) ===" && ls -la /app/data/ 2>/dev/null || echo "No /app/data directory"
RUN echo "=== Looking for CSV files ===" && find /app -name "*.csv" -type f 2>/dev/null || echo "No CSV files found"

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]