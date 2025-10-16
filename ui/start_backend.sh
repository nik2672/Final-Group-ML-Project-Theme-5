#!/bin/bash

# Start the FastAPI backend server

cd "$(dirname "$0")/backend"

echo "Starting FastAPI backend on http://localhost:8000"
python main.py
