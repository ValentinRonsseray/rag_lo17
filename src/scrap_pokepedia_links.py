import requests
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
import logging
from bs4 import BeautifulSoup
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des limites
MAX_PAGES = 50  # Nombre maximum de pages à récupérer
BATCH_SIZE = 5  # Nombre de pages par lot
REQUEST_DELAY = 3  # Délai entre les requêtes en secondes

def get_pokepedia_urls(max_pages: int = MAX_PAGES) -> List[str]:
    """Récupère la liste des URLs des pages principales de Poképedia.
    
    Args:
        max_pages: Nombre maximum de pages à récupérer
    """
    api_url = "https://www.pokepedia.fr/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": str(min(500, max_pages)),  # Limite le nombre de pages par requête
        "apnamespace": "0"  # Uniquement les pages principales
    }
    
    all_urls = []
    continue_token = None
    
    while len(all_urls) < max_pages:
        if continue_token:
            params["apcontinue"] = continue_token
            
        try:
            response = requests.get(api_url, params=params)
            data = response.json()
            
            pages = data.get("query", {}).get("allpages", [])
            for page in pages:
                if len(all_urls) >= max_pages:
                    break
                    
                title = page["title"]
                # Ne garde que les pages de Pokémon (commençant par "Pokémon")
                if title.startswith("Pokémon"):
                    url = f"https://www.pokepedia.fr/{title.replace(' ', '_')}"
                    all_urls.append(url)
            
            if "continue" in data and len(all_urls) < max_pages:
                continue_token = data["continue"]["apcontinue"]
                time.sleep(REQUEST_DELAY)  # Pause entre les requêtes API
            else:
                break
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des URLs: {str(e)}")
            break
    
    logger.info(f"Nombre d'URLs trouvées: {len(all_urls)}")
    return all_urls

def clean_html_content(html_content: str) -> str:
    """Nettoie le contenu HTML pour ne garder que le texte pertinent."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Suppression des éléments non pertinents
    for element in soup.find_all(['script', 'style', 'table', 'div', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Extraction du texte principal
    text = soup.get_text(separator=' ', strip=True)
    return text

def retrieve_pokepedia_documents(max_pages: int = MAX_PAGES) -> List[Document]:
    """Récupère les pages Poképedia et les convertit en documents pour le RAG.
    
    Args:
        max_pages: Nombre maximum de pages à récupérer
    """
    # Récupération des URLs
    urls = get_pokepedia_urls(max_pages)
    logger.info(f"Nombre d'URLs trouvées: {len(urls)}")
    
    # Chargement des documents par lots
    all_documents = []
    
    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i:i + BATCH_SIZE]
        logger.info(f"Traitement du lot {i//BATCH_SIZE + 1}/{(len(urls) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        try:
            # Utilisation de WebBaseLoader pour charger les pages
            loader = WebBaseLoader(
                web_paths=batch_urls,
                verify_ssl=False,
                custom_html_parser=clean_html_content
            )
            documents = loader.load()
            
            # Ajout des métadonnées
            for doc in documents:
                doc.metadata.update({
                    "source": "pokepedia",
                    "url": doc.metadata.get("source", "")
                })
            
            all_documents.extend(documents)
            
            # Pause entre les lots
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du lot {i//BATCH_SIZE + 1}: {str(e)}")
            continue
    
    logger.info(f"Nombre total de documents récupérés: {len(all_documents)}")
    return all_documents

if __name__ == "__main__":
    documents = retrieve_pokepedia_documents()
    print(f"Nombre de documents récupérés: {len(documents)}")
