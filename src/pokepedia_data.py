"""
Module pour gérer les données Poképédia et enrichir les informations des Pokémon.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class PokepediaData:
    """Gestionnaire des données Poképédia."""

    def __init__(self, data_dir: str = "data/pokepedia"):
        """Initialise le gestionnaire de données Poképédia.

        Args:
            data_dir: Répertoire contenant les données Poképédia
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pokemon_data = {}
        self.load_data()

    def load_data(self):
        """Charge les données Poképédia depuis les fichiers JSON."""
        for file_path in self.data_dir.glob("*.json"):
            pokemon_name = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.pokemon_data[pokemon_name] = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des données pour {pokemon_name}: {e}")

    def get_pokemon_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations Poképédia pour un Pokémon.

        Args:
            name: Nom du Pokémon

        Returns:
            Dictionnaire contenant les informations Poképédia ou None si non trouvé
        """
        return self.pokemon_data.get(name.lower())

    def enrich_pokemon_document(self, pokemon: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les données d'un Pokémon avec les informations Poképédia.

        Args:
            pokemon: Données du Pokémon à enrichir

        Returns:
            Données enrichies du Pokémon
        """
        name = pokemon.get("name", "").lower()
        pokepedia_info = self.get_pokemon_info(name)

        if pokepedia_info:
            # Les données peuvent provenir d'un scraping simple (champ "content")
            # ou être déjà structurées. On harmonise pour le reste du code.
            pk_data: Dict[str, Any] = {}

            # Texte brut unique
            if "content" in pokepedia_info:
                pk_data["description"] = pokepedia_info.get("content", "")

            # Champs structurés si présents
            for key in [
                "description",
                "biology",
                "behavior",
                "habitat",
                "trivia",
                "evolution",
                "mythology",
            ]:
                if pokepedia_info.get(key):
                    pk_data[key] = pokepedia_info[key]

            pokemon["pokepedia"] = pk_data

        return pokemon


# Données d'exemple pour Arcanin
ARCANINE_DATA = {
    "description": "Son port altier et son attitude fière ont depuis longtemps conquis le cœur des hommes.",
    "biology": "Arcanin est un Pokémon majestueux qui ressemble à un grand chien. Son pelage est principalement orange avec des marques noires et blanches. Il possède une crinière imposante et une queue touffue.",
    "behavior": "Arcanin est connu pour sa loyauté et son courage. Il est extrêmement fidèle à son dresseur et protégera son territoire avec ferveur. Malgré son apparence imposante, il est doux avec les personnes qu'il apprécie.",
    "habitat": "On trouve Arcanin dans les plaines et les forêts. Il préfère les endroits ouverts où il peut courir librement.",
    "trivia": [
        "Arcanin est basé sur le chien-lion chinois, une créature mythologique.",
        "Dans la mythologie japonaise, Arcanin est considéré comme un symbole de loyauté et de courage.",
        "Son nom vient de la combinaison de 'arc' (arc-en-ciel) et 'canine'.",
    ],
    "evolution": "Arcanin est l'évolution de Caninos. Il évolue grâce à une Pierre de Feu.",
    "mythology": "Dans la culture Pokémon, Arcanin est souvent associé aux légendes de loyauté et de bravoure. On raconte que les premiers Arcanin étaient les gardiens des temples anciens.",
}


def initialize_pokepedia_data():
    """Initialise les données Poképédia avec des exemples."""
    data_dir = Path("data/pokepedia")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des données d'exemple pour Arcanin
    with open(data_dir / "arcanine.json", "w", encoding="utf-8") as f:
        json.dump(ARCANINE_DATA, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    initialize_pokepedia_data()
