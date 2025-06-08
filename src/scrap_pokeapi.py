import requests
import json
import os
import time
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://pokeapi.co/api/v2"
DATA_DIR = "data/pokeapi"
REQUEST_DELAY = 0.1  # Délai entre les requêtes en secondes
MAX_WORKERS = 5  # Nombre de workers pour les requêtes parallèles

# Création du dossier de données
os.makedirs(DATA_DIR, exist_ok=True)

def get_base_pokemon_name(name: str) -> str:
    """Extrait le nom de base du Pokémon (sans forme alternative)."""
    # Liste des suffixes connus pour les formes alternatives
    suffixes = [
        '-mega', '-mega-x', '-mega-y', '-alola', '-galar', '-hisui',
        '-gmax', '-eternamax', '-ash', '-power-construct', '-complete',
        '-10', '-50', '-100', '-therian', '-incarnate', '-land', '-sky',
        '-ordinary', '-aria', '-baile', '-midday', '-midnight', '-dusk',
        '-dawn', '-shield', '-solo', '-school', '-red-striped', '-blue-striped',
        '-east', '-west', '-fan', '-frost', '-heat', '-mow', '-wash',
        '-normal', '-plant', '-sandy', '-trash', '-overcast', '-sunny',
        '-rainy', '-snowy', '-attack', '-defense', '-speed'
    ]
    
    base_name = name
    for suffix in suffixes:
        if name.endswith(suffix):
            base_name = name[:-len(suffix)]
            break
    
    return base_name

def get_pokemon_list() -> List[Dict[str, Any]]:
    """Récupère la liste de tous les Pokémon."""
    logger.info("Récupération de la liste des Pokémon...")
    url = f"{BASE_URL}/pokemon?limit=2000"  # Limite élevée pour avoir tous les Pokémon
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Nombre de Pokémon trouvés: {len(data['results'])}")
        return data['results']
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des Pokémon: {str(e)}", exc_info=True)
        return []

def get_pokemon_details(pokemon: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère les détails d'un Pokémon."""
    name = pokemon['name']
    url = pokemon['url']
    logger.debug(f"Récupération des détails pour {name}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Récupération des informations supplémentaires
        base_name = get_base_pokemon_name(name)
        species_data = get_pokemon_species(base_name)
        if species_data:
            data['species_info'] = species_data
            data['base_form'] = base_name
            
        return data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails pour {name}: {str(e)}", exc_info=True)
        return {}

def get_pokemon_species(name: str) -> Dict[str, Any]:
    """Récupère les informations sur l'espèce d'un Pokémon."""
    logger.debug(f"Récupération des informations sur l'espèce pour {name}")
    
    try:
        url = f"{BASE_URL}/pokemon-species/{name}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations sur l'espèce pour {name}: {str(e)}", exc_info=True)
        return {}

def save_pokemon_data(pokemon_data: Dict[str, Any], name: str):
    """Sauvegarde les données d'un Pokémon dans un fichier JSON."""
    if not pokemon_data:
        return
        
    file_path = os.path.join(DATA_DIR, f"{name}.json")
    logger.debug(f"Sauvegarde des données pour {name} dans {file_path}")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pokemon_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Données sauvegardées pour {name}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données pour {name}: {str(e)}", exc_info=True)

def process_pokemon(pokemon: Dict[str, Any]) -> bool:
    """Traite un Pokémon (récupération des détails et sauvegarde)."""
    name = pokemon['name']
    logger.info(f"Traitement de {name}")
    
    # Vérification si les données existent déjà
    file_path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(file_path):
        logger.info(f"Les données pour {name} existent déjà")
        return True
    
    # Récupération et sauvegarde des données
    pokemon_data = get_pokemon_details(pokemon)
    if pokemon_data:
        save_pokemon_data(pokemon_data, name)
        time.sleep(REQUEST_DELAY)  # Pause entre les requêtes
        return True
    return False

def main():
    """Fonction principale."""
    logger.info("Début du scraping des données Pokémon")
    
    # Récupération de la liste des Pokémon
    pokemon_list = get_pokemon_list()
    if not pokemon_list:
        logger.error("Impossible de récupérer la liste des Pokémon")
        return
    
    # Traitement parallèle des Pokémon
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pokemon = {
            executor.submit(process_pokemon, pokemon): pokemon
            for pokemon in pokemon_list
        }
        
        for future in as_completed(future_to_pokemon):
            pokemon = future_to_pokemon[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {pokemon['name']}: {str(e)}", exc_info=True)
                failed += 1
    
    logger.info(f"Scraping terminé. Succès: {successful}, Échecs: {failed}")

if __name__ == "__main__":
    main()
