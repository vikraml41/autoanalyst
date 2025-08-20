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
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]