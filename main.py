"""
Script principal pour le système RAG Pokémon.
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

def run_command(command: str) -> bool:
    """Exécute une commande shell et retourne True si elle réussit."""
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande '{command}': {e}")
        return False

def setup_environment():
    """Configure l'environnement pour l'application."""
    print("Configuration de l'environnement...")
    
    # Vérifier que Python 3.12 est installé
    if sys.version_info < (3, 12):
        print("Python 3.12 ou supérieur est requis.")
        return False
    
    # Vérifier le fichier .env
    if not Path(".env").exists():
        print("Erreur : Fichier .env non trouvé")
        print("Veuillez créer un fichier .env avec votre GOOGLE_API_KEY")
        return False
    
    # Charger les variables d'environnement
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Erreur : GOOGLE_API_KEY non trouvée dans le fichier .env")
        return False
    
    # Installer les dépendances
    if not run_command("pip install -r requirements.txt"):
        return False
    
    return True

def setup_directories():
    """Crée les dossiers nécessaires."""
    print("\nCréation des dossiers nécessaires...")
    
    directories = [
        "data/uploads",
        "data/pokeapi",
        "data/indexes",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Dossier créé/vérifié : {directory}")

def check_and_scrape_data():
    """Vérifie et récupère les données si nécessaire."""
    print("\nVérification des données...")
    
    # Vérifier si les données PokéAPI existent
    if not any(Path("data/pokeapi").glob("*")):
        print("Récupération des données PokéAPI...")
        if not run_command("python src/scrap_pokeapi.py"):
            return False
    
    # Vérifier si les index existent
    if not Path("data/indexes/type_index.json").exists():
        print("Construction des index...")
        if not run_command("python src/build_pokemon_index.py"):
            return False
    
    return True

def run_application():
    """Lance l'application Streamlit."""
    print("\nLancement de l'application...")
    return run_command("streamlit run app.py")

def main():
    """Fonction principale."""
    print("=== Démarrage du système RAG Pokémon ===")
    print(f"Date et heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration de l'environnement
    if not setup_environment():
        print("Échec de la configuration de l'environnement.")
        return
    
    # Création des dossiers
    setup_directories()
    
    # Vérification et récupération des données
    if not check_and_scrape_data():
        print("Échec de la vérification/récupération des données.")
        return
    
    # Lancement de l'application
    if not run_application():
        print("Échec du lancement de l'application.")
        return
    
    print("\n=== Application lancée avec succès ===")

if __name__ == "__main__":
    main() 