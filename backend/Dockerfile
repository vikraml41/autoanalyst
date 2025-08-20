# Use Python 3.9 with full OS (not slim) for compilation tools
FROM python:3.9

# Install system libraries needed for scipy/sklearn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# First install numpy (scipy depends on it)
RUN pip install --upgrade pip
RUN pip install numpy==1.24.3

# Then install scipy with proper flags
RUN pip install scipy --no-cache-dir

# Then install scikit-learn
RUN pip install scikit-learn

# Copy requirements file
COPY requirements.txt .

# Install all other packages
RUN pip install -r requirements.txt

# Copy all backend files
COPY . .

# Command to run the app
# Use official Python runtime with Debian (has better scipy support)
FROM python:3.9-slim-bullseye

# Install system dependencies for scipy/sklearn compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-dev \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for scipy
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install numpy first (critical for scipy)
RUN pip install numpy==1.24.3

# Install scipy with specific version that works
RUN pip install scipy==1.10.1 --no-cache-dir

# Install scikit-learn
RUN pip install scikit-learn==1.3.0 --no-cache-dir

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files INCLUDING your CSV data
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Run the app
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]