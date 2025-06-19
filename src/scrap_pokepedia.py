import os
import time
import json
from typing import List
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pokepedia.fr"
API_URL = f"{BASE_URL}/api.php"
DATA_DIR = "data/pokepedia"
REQUEST_DELAY = 0.5
MAX_PAGES = 50


def get_pokemon_urls(max_pages: int = MAX_PAGES) -> List[str]:
    """Retrieve Pokémon page URLs from Poképedia."""
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": "0",
    }
    urls: List[str] = []
    token = None
    while len(urls) < max_pages:
        if token:
            params["apcontinue"] = token
        try:
            resp = requests.get(API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"Erreur API Poképédia: {exc}")
            break
        pages = data.get("query", {}).get("allpages", [])
        for page in pages:
            if len(urls) >= max_pages:
                break
            title = page.get("title", "")
            if title.startswith("Pokémon"):
                slug = title.replace(" ", "_")
                urls.append(f"{BASE_URL}/{slug}")
        token = data.get("continue", {}).get("apcontinue")
        if not token:
            break
        time.sleep(REQUEST_DELAY)
    return urls[:max_pages]


def clean_html(html: str) -> str:
    """Extract text from raw HTML, removing scripts and style."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "footer", "nav", "header", "table"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def fetch_page(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return clean_html(resp.text)


def save_content(name: str, url: str, content: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name.lower()}.json")
    data = {"url": url, "content": content, "timestamp": time.time()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scrape_pokepedia(max_pages: int = MAX_PAGES):
    urls = get_pokemon_urls(max_pages)
    for url in urls:
        name = url.split("/")[-1]
        try:
            text = fetch_page(url)
            save_content(name, url, text)
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            print(f"Erreur lors de la récupération de {url}: {exc}")


if __name__ == "__main__":
    scrape_pokepedia()
