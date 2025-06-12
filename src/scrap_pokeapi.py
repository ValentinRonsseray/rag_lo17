import requests
import json
import os
import time
import logging
from typing import Dict, List, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------
# Configuration du logging
# ---------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------
# Constantes de configuration
# ---------------------------------
BASE_URL = "https://pokeapi.co/api/v2"
DATA_DIR = "data/pokeapi"
REQUEST_DELAY = 0.1          # délai entre les requêtes en secondes
MAX_WORKERS = 5              # nombre de workers pour les requêtes parallèles
GENERATION_ID = 1            # *** On ne garde QUE les Pokémon de la génération 4 ***

# Création du dossier de sortie
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def get_base_pokemon_name(name: str) -> str:
    """Extrait le nom de base du Pokémon (sans forme alternative)."""
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
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def get_generation_pokemon_names(gen_id: int) -> Set[str]:
    """Retourne l'ensemble des noms d'espèces appartenant à une génération donnée."""
    logger.info(f"Récupération des Pokémon de la génération {gen_id}…")
    url = f"{BASE_URL}/generation/{gen_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        names = {species["name"] for species in data.get("pokemon_species", [])}
        logger.info(f"{len(names)} espèces trouvées pour la génération {gen_id}.")
        return names
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la génération {gen_id}: {str(e)}", exc_info=True)
        return set()


def get_pokemon_list() -> List[Dict[str, Any]]:
    """Récupère la liste brute de tous les Pokémon (formes comprises)."""
    logger.info("Récupération de la liste complète des Pokémon…")
    url = f"{BASE_URL}/pokemon?limit=2000"  # suffisamment grand pour tout récupérer
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results']
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des Pokémon: {str(e)}", exc_info=True)
        return []

# ---------------------------------------------------------------------------
# Récupération et nettoyage des données détaillées
# ---------------------------------------------------------------------------

def strip_urls_from_dict(data: Dict[str, Any]):
    """Supprime récursivement les clés contenant 'url' dans un dictionnaire ou une liste."""
    if isinstance(data, dict):
        for key in list(data.keys()):
            if 'url' in key.lower():
                del data[key]
            else:
                strip_urls_from_dict(data[key])
    elif isinstance(data, list):
        for item in data:
            strip_urls_from_dict(item)


def get_pokemon_species(name: str) -> Dict[str, Any]:
    """Récupère les informations sur l'espèce (species) d'un Pokémon."""
    url = f"{BASE_URL}/pokemon-species/{name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # -------------------------
        # Filtrage des informations
        # -------------------------
        filtered: Dict[str, Any] = {}

        # On garde des champs simples et utiles
        for key in [
            'id', 'name', 'order', 'gender_rate', 'capture_rate', 'base_happiness',
            'is_baby', 'is_legendary', 'is_mythical', 'hatch_counter',
            'has_gender_differences', 'forms_switchable', 'growth_rate',
            'color', 'habitat'  # rajouté ici pour simplifier
        ]:
            if key in data:
                filtered[key] = data[key]

        # Noms, descriptions, genres : seulement EN / FR / JA
        if 'names' in data:
            filtered['names'] = [n for n in data['names'] if n['language']['name'] in {'en', 'fr', 'ja'}]
        if 'flavor_text_entries' in data:
            filtered['flavor_text_entries'] = [f for f in data['flavor_text_entries'] if f['language']['name'] in {'en', 'fr', 'ja'}]
        if 'genera' in data:
            filtered['genera'] = [g for g in data['genera'] if g['language']['name'] in {'en', 'fr', 'ja'}]

        # Chaîne d'évolution : on ne garde que l'URL (sera supprimée plus tard par strip_urls_from_dict)
        if 'evolution_chain' in data:
            filtered['evolution_chain'] = data['evolution_chain']

        # Suppression des URLs restantes (y compris dans evolution_chain)
        strip_urls_from_dict(filtered)
        return filtered
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'espèce pour {name}: {str(e)}", exc_info=True)
        return {}


def get_pokemon_details(pokemon: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère les détails d'un Pokémon donné (forme ou non)."""
    name = pokemon['name']
    url = pokemon['url']
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Nettoyage : on enlève les champs très verbeux
        for noisy_field in ['moves', 'location_area_encounters', 'sprites']:
            if noisy_field in data:
                del data[noisy_field]

        # Nettoyage récursif des URLs
        strip_urls_from_dict(data)

        # Infos supplémentaires issues de l'espèce
        base_name = get_base_pokemon_name(name)
        species_info = get_pokemon_species(base_name)
        if species_info:
            data['species_info'] = species_info
            data['base_form'] = base_name

        return data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails pour {name}: {str(e)}", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Sauvegarde locale
# ---------------------------------------------------------------------------

def save_pokemon_data(content: Dict[str, Any], name: str):
    if not content:
        return
    path = os.path.join(DATA_DIR, f"{name}.json")
    try:
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(content, fp, ensure_ascii=False, indent=2)
        logger.debug(f"Fichier sauvegardé : {path}")
    except Exception as e:
        logger.error(f"Erreur de sauvegarde pour {name}: {str(e)}", exc_info=True)


# ---------------------------------------------------------------------------
# Traitement parallèle d'un Pokémon
# ---------------------------------------------------------------------------

def process_pokemon(pokemon: Dict[str, Any]) -> bool:
    name = pokemon['name']
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        logger.debug(f"{name} déjà présent, on passe.")
        return True

    details = get_pokemon_details(pokemon)
    if details:
        save_pokemon_data(details, name)
        time.sleep(REQUEST_DELAY)
        return True
    return False

# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    logger.info("--- Lancement du scraping Pokémon (Génération 4 uniquement) ---")

    allowed_species = get_generation_pokemon_names(GENERATION_ID)
    if not allowed_species:
        logger.error("Liste d'espèces vide. Arrêt du programme.")
        return

    # Liste complète (avec formes), puis filtrage
    all_pokemon = get_pokemon_list()
    pokemon_list = [p for p in all_pokemon if get_base_pokemon_name(p['name']) in allowed_species]
    logger.info(f"{len(pokemon_list)} formes à traiter après filtrage.")

    success = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pokemon, p): p for p in pokemon_list}
        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
                failed += 1

    logger.info(f"--- Terminé. Succès: {success} | Échecs: {failed} ---")


if __name__ == "__main__":
    main()
