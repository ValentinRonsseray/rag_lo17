#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please create a .env file with your GOOGLE_API_KEY"
    exit 1
fi

# Create necessary directories
mkdir -p data/uploads chroma_db data/pokeapi data/indexes

# Scrape PokéAPI data if not already present
if [ -z "$(ls -A data/pokeapi 2>/dev/null)" ]; then
    echo "Scraping PokéAPI data..."
    python src/scrap_pokeapi.py
fi

# Build indexes if they do not exist
if [ ! -f data/indexes/type_index.json ]; then
    echo "Building indexes..."
    python src/build_pokemon_index.py
fi

# Run the application
streamlit run app.py
