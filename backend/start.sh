#!/bin/bash
# Start script for Render.com backend deployment

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting FastAPI server..."
python api_server.py
