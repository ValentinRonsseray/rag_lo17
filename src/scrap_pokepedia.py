import os
import time
import json
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pokepedia.fr"
API_URL = f"{BASE_URL}/api.php"
DATA_DIR = "data/pokepedia"
REQUEST_DELAY = 0.5
MAX_PAGES = None  # Parcours complet par défaut
AP_NAMESPACE = "120"  # pages Pokémon uniquement


def get_all_page_titles(limit: Optional[int] = MAX_PAGES) -> List[str]:
    """Récupère la liste de toutes les pages du namespace Pokémon (120)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": AP_NAMESPACE,
    }
    titles: List[str] = []
    token: Optional[str] = None

    while True:
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
            titles.append(page.get("title", ""))
            if limit and len(titles) >= limit:
                break

        if limit and len(titles) >= limit:
            break

        token = data.get("continue", {}).get("apcontinue")
        if not token:
            break

        time.sleep(REQUEST_DELAY)

    return titles[:limit] if limit else titles


def title_to_url(title: str) -> str:
    slug = title.replace(" ", "_")
    return f"{BASE_URL}/{slug}"


def extract_paragraphs(html: str) -> str:
    """Extrait les paragraphes pertinents d'une page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "footer", "nav", "header", "table"]):
        tag.decompose()

    content = soup.find("div", class_="mw-parser-output") or soup
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in content.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n\n".join(paragraphs)


def fetch_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return extract_paragraphs(resp.text)


def save_content(name: str, url: str, content: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name.lower()}.json")
    data = {"url": url, "content": content, "timestamp": time.time()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scrape_pokepedia(max_pages: int = MAX_PAGES):
    titles = get_all_page_titles(max_pages)
    for title in titles:
        url = title_to_url(title)
        try:
            text = fetch_page(url)
        except Exception as exc:
            print(f"Erreur lors de la récupération de {url}: {exc}")
            continue

        save_content(title, url, text)
        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    scrape_pokepedia()
