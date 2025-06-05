#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please create a .env file with your GOOGLE_API_KEY"
    exit 1
fi

# Create necessary directories
mkdir -p data/uploads chroma_db

# Run the application
streamlit run app.py

docker build -t rag-qa .
docker run -p 8501:8501 -e GOOGLE_API_KEY='your-api-key' rag-qa 