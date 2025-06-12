import requests
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
import logging
from bs4 import BeautifulSoup
import time
import os
import hashlib
import json

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration des limites
MAX_PAGES = 50  # Nombre maximum de pages à récupérer
BATCH_SIZE = 5  # Nombre de pages par lot
REQUEST_DELAY = 0.1  # Délai entre les requêtes en secondes
CACHE_DIR = "data/pokepedia"  # Dossier de cache

# Création du dossier de cache s'il n'existe pas
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(url: str) -> str:
    """Génère le chemin du fichier de cache pour une URL donnée."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{url_hash}.json")
    logger.debug(f"Chemin de cache généré pour {url}: {cache_path}")
    return cache_path

def save_to_cache(url: str, content: str, metadata: dict):
    """Sauvegarde le contenu d'une page dans le cache."""
    cache_path = get_cache_path(url)
    logger.debug(f"Sauvegarde dans le cache: {url}")
    data = {
        "url": url,
        "content": content,
        "metadata": metadata,
        "timestamp": time.time()
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Sauvegarde réussie dans {cache_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde dans le cache: {str(e)}", exc_info=True)

def load_from_cache(url: str) -> tuple:
    """Charge le contenu d'une page depuis le cache."""
    cache_path = get_cache_path(url)
    logger.debug(f"Tentative de chargement depuis le cache: {url}")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Chargement réussi depuis {cache_path}")
            return data["content"], data["metadata"]
        except Exception as e:
            logger.error(f"Erreur lors du chargement depuis le cache: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Pas de cache trouvé pour {url}")
    return None, None

def get_pokepedia_urls(max_pages: int = MAX_PAGES) -> List[str]:
    """Récupère la liste des URLs des pages principales de Poképedia."""
    logger.debug(f"Début de la récupération des URLs (max_pages={max_pages})")
    api_url = "https://www.pokepedia.fr/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": str(min(500, max_pages)),
        "apnamespace": "0"
    }
    
    all_urls = []
    continue_token = None
    request_count = 0
    
    while len(all_urls) < max_pages:
        request_count += 1
        logger.debug(f"Requête API #{request_count} - URLs trouvées: {len(all_urls)}/{max_pages}")
        
        if continue_token:
            params["apcontinue"] = continue_token
            logger.debug(f"Utilisation du token de continuation: {continue_token[:20]}...")
            
        try:
            logger.debug(f"Envoi de la requête à l'API avec les paramètres: {params}")
            response = requests.get(api_url, params=params)
            data = response.json()
            
            pages = data.get("query", {}).get("allpages", [])
            logger.debug(f"Nombre de pages reçues dans la réponse: {len(pages)}")
            
            for page in pages:
                if len(all_urls) >= max_pages:
                    break
                    
                title = page["title"]
                if title.startswith("Pokémon"):
                    url = f"https://www.pokepedia.fr/{title.replace(' ', '_')}"
                    all_urls.append(url)
                    logger.debug(f"Ajout de l'URL: {url}")
            
            if "continue" in data and len(all_urls) < max_pages:
                continue_token = data["continue"]["apcontinue"]
                logger.debug(f"Pause de {REQUEST_DELAY} secondes avant la prochaine requête")
                time.sleep(REQUEST_DELAY)
            else:
                logger.debug("Plus de pages à récupérer ou limite atteinte")
                break
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des URLs: {str(e)}", exc_info=True)
            break
    
    logger.info(f"Récupération terminée. Nombre total d'URLs trouvées: {len(all_urls)}")
    return all_urls

def clean_html_content(html_content: str) -> str:
    """Nettoie le contenu HTML pour ne garder que le texte pertinent."""
    logger.debug("Début du nettoyage du contenu HTML")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Suppression des éléments non pertinents
    elements_removed = 0
    for element in soup.find_all(['script', 'style', 'table', 'div', 'nav', 'footer', 'header']):
        element.decompose()
        elements_removed += 1
    
    logger.debug(f"Nombre d'éléments HTML supprimés: {elements_removed}")
    
    # Extraction du texte principal
    text = soup.get_text(separator=' ', strip=True)
    logger.debug(f"Longueur du texte extrait: {len(text)} caractères")
    return text

def retrieve_pokepedia_documents(max_pages: int = MAX_PAGES) -> List[Document]:
    """Récupère les pages Poképedia et les convertit en documents pour le RAG."""
    logger.info(f"Début de la récupération des documents (max_pages={max_pages})")
    
    # Récupération des URLs
    urls = get_pokepedia_urls(max_pages)
    logger.info(f"Nombre d'URLs trouvées: {len(urls)}")
    
    # Chargement des documents par lots
    all_documents = []
    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(urls), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_urls = urls[i:i + BATCH_SIZE]
        logger.info(f"Traitement du lot {batch_num}/{total_batches} ({len(batch_urls)} URLs)")
        
        try:
            batch_documents = []
            for url in batch_urls:
                logger.debug(f"Traitement de l'URL: {url}")
                
                # Vérification du cache
                cached_content, cached_metadata = load_from_cache(url)
                
                if cached_content:
                    logger.info(f"Chargement depuis le cache: {url}")
                    doc = Document(
                        page_content=cached_content,
                        metadata=cached_metadata or {"source": "pokepedia", "url": url}
                    )
                    batch_documents.append(doc)
                else:
                    logger.info(f"Chargement depuis le web: {url}")
                    try:
                        loader = WebBaseLoader(
                            web_paths=[url],
                            verify_ssl=False,
                            custom_html_parser=clean_html_content
                        )
                        docs = loader.load()
                        
                        if docs:
                            doc = docs[0]
                            doc.metadata.update({
                                "source": "pokepedia",
                                "url": url
                            })
                            save_to_cache(url, doc.page_content, doc.metadata)
                            batch_documents.append(doc)
                        else:
                            logger.warning(f"Aucun document chargé pour {url}")
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement de {url}: {str(e)}", exc_info=True)
            
            all_documents.extend(batch_documents)
            logger.info(f"Lot {batch_num} terminé. Documents dans ce lot: {len(batch_documents)}")
            
            if batch_num < total_batches:
                logger.debug(f"Pause de {REQUEST_DELAY} secondes avant le prochain lot")
                time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du lot {batch_num}: {str(e)}", exc_info=True)
            continue
    
    logger.info(f"Récupération terminée. Nombre total de documents: {len(all_documents)}")
    return all_documents

if __name__ == "__main__":
    documents = retrieve_pokepedia_documents()
    print(f"Nombre de documents récupérés: {len(documents)}")
