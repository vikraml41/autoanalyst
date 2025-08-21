# Use official Jupyter image with scipy pre-installed
FROM jupyter/scipy-notebook:python-3.9.13

# Switch to root to install packages
USER root

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install additional packages (scipy/sklearn already in base image!)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
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

# Copy all your backend files
COPY . .

# Create data directory for CSV files
RUN mkdir -p /app/data

# Set permissions
RUN chown -R jovyan:users /app

# Switch back to non-root user
USER jovyan

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]