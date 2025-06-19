import os
import time
import json
from typing import List, Optional, Tuple
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pokepedia.fr"
DATA_DIR = "data/pokepedia"
REQUEST_DELAY = 0.1
MAX_PAGES = None  # Parcours complet par défaut

# Catégorie des Pokémon de la première génération
CATEGORY_URL = (
    f"{BASE_URL}/Cat%C3%A9gorie:Pok%C3%A9mon_de_la_premi%C3%A8re_g%C3%A9n%C3%A9ration"
)


def get_category_links(limit: Optional[int] = MAX_PAGES) -> List[Tuple[str, str]]:
    """Récupère les liens de la catégorie des Pokémon de première génération."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(CATEGORY_URL, headers=headers)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Erreur lors de la récupération de la catégorie: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.select("div.mw-category a")
    links: List[Tuple[str, str]] = []
    seen = set()
    for a in anchors:
        name = a.get_text(strip=True)
        href = a.get("href", "")
        if not name or not href or name in seen:
            continue
        seen.add(name)
        if not href.startswith("http"):
            href = BASE_URL + href
        links.append((name, href))
        if limit and len(links) >= limit:
            break

    return links

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
    links = get_category_links(max_pages)
    for name, url in links:
        try:
            text = fetch_page(url)
        except Exception as exc:
            print(f"Erreur lors de la récupération de {url}: {exc}")
            continue

        save_content(name, url, text)
        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    scrape_pokepedia()
