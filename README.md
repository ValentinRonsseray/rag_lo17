# RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system built with LangChain, Google's Gemini model, and ChromaDB.

## Features

- Document ingestion and embedding using ChromaDB
- Question answering using Google's Gemini model
- Evaluation metrics (Exact Match, F1 Score, Faithfulness)
- Hallucination detection and logging
- Streamlit web interface
- Docker support for easy deployment

## Prerequisites

- Python 3.9 or higher
- Google API key for Gemini model
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google API key:
```bash
export GOOGLE_API_KEY='your-api-key'
```

## Usage

### Running Locally

1. Start the application:
```bash
./run_app.sh
```
The script checks if Pokémon data and indexes are present. If not,
it automatically downloads the data from **PokéAPI** and scrapes
Poképedia. The scraper starts from the category page for the first
generation Pokémon, downloads every linked page, and saves its
paragraphs under `data/pokepedia` before the indexes are built.

2. Open your browser and navigate to `http://localhost:8501`

### Using Docker

1. Build the Docker image:
```bash
docker build -t rag-qa .
```

2. Run the container:
```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY='your-api-key' rag-qa
```

## Project Structure

```
.
├── app.py                 # Streamlit application
├── src/
│   ├── rag_core.py       # Core RAG components
│   └── evaluation.py     # Evaluation metrics
│   ├── scrap_pokeapi.py  # Download data from PokéAPI
│   └── scrap_pokepedia.py # Scrape texts from Poképedia
├── data/
│   └── uploads/          # Uploaded documents
├── chroma_db/            # Vector store
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── run_app.sh           # Startup script
```

## Evaluation

The system includes several evaluation metrics:
- Exact Match Score
- F1 Score
- Faithfulness Score (LLM-based)

Results are saved in `eval_metrics.csv` and visualized in `evaluation_metrics.png`.
The `evaluate_rag.py` script includes a small dataset of eight sample
questions with their expected answers in order to benchmark the system.
Feel free to adapt these examples to suit your needs.

## Hallucination Detection

The system detects potential hallucinations when the faithfulness score is below 0.7. Detected cases are logged in `hallucinations.csv`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
