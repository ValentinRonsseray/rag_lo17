"""
python main.py pour configurer l'environnement
python main.py --run pour lancer l'application
"""
import os
import sys
import asyncio
import subprocess
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import argparse


def run_command(command: str) -> bool:
    """exécute une commande shell et retourne true si elle réussit."""
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"erreur lors de l'exécution de la commande '{command}': {e}")
        return False


def setup_environment():
    """configure l'environnement pour l'application."""
    print("configuration de l'environnement...")

    # vérifier que python 3.12 est installé
    if sys.version_info < (3, 12):
        print("python 3.12 ou supérieur est requis.")
        return False

    # vérifier le fichier .env
    if not Path(".env").exists():
        print("erreur : fichier .env non trouvé")
        print("veuillez créer un fichier .env avec votre google_api_key")
        return False

    # charger les variables d'environnement
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("erreur : google_api_key non trouvée dans le fichier .env")
        return False

    # installer les dépendances
    if not run_command("pip install -r requirements.txt"):
        return False

    return True


def setup_directories():
    """crée les dossiers nécessaires."""
    print("\ncréation des dossiers nécessaires...")

    directories = ["data/uploads", "data/pokeapi", "data/indexes", "chroma_db"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"dossier créé/vérifié : {directory}")


def check_and_scrape_data():
    """vérifie et récupère les données si nécessaire."""
    print("\nvérification des données...")

    # vérifier si les données pokéapi existent
    if not any(Path("data/pokeapi").glob("*")):
        print("récupération des données pokéapi...")
        if not run_command("python src/scrap_pokeapi.py"):
            return False

    # vérifier si les données poképédia existent
    if not any(Path("data/pokepedia").glob("*.json")):
        print("récupération des données poképédia...")
        if not run_command("python src/scrap_pokepedia.py"):
            return False

    # vérifier si les index existent
    if not Path("data/indexes/type_index.json").exists():
        print("construction des index...")
        if not run_command("python src/build_pokemon_index.py"):
            return False

    return True


def run_application():
    """lance l'application streamlit."""
    print("\nlancement de l'application...")
    return run_command("streamlit run app.py")


def main():
    """fonction principale."""
    # parse les arguments
    parser = argparse.ArgumentParser(description="système rag pokémon")
    parser.add_argument(
        "--run", action="store_true", help="lance directement l'application"
    )
    args = parser.parse_args()

    print("=== démarrage du système rag pokémon ===")
    print(f"date et heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # configuration de l'environnement
    if not setup_environment():
        print("échec de la configuration de l'environnement.")
        return

    # création des dossiers
    setup_directories()

    # vérification et récupération des données
    if not check_and_scrape_data():
        print("échec de la vérification/récupération des données.")
        return

    # si --run est spécifié, lance l'application
    if args.run:
        if not run_application():
            print("échec du lancement de l'application.")
            return
        print("\n=== application lancée avec succès ===")
    else:
        print("\n=== configuration terminée avec succès ===")
        print("pour lancer l'application, utilisez : python main.py --run")


if __name__ == "__main__":
    main()
