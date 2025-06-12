"""
Script pour construire des index inverses des Pokémon basés sur leurs caractéristiques.
(Remarque : les chaînes d'évolution ne sont plus indexées comme demandé.)
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict


def load_pokemon_data() -> List[Dict[str, Any]]:
    """Charge les données Pokémon depuis les fichiers JSON présents dans data/pokeapi."""
    data_dir = "data/pokeapi"
    pokemon_data: List[Dict[str, Any]] = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                pokemon_data.append(json.load(f))

    return pokemon_data


# ---------------------------------------------------------------------------
# Index par type
# ---------------------------------------------------------------------------

def build_type_index(pokemon_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Construit un index des Pokémon par type principal / secondaire."""
    type_index: Dict[str, List[str]] = defaultdict(list)

    for pokemon in pokemon_data:
        name = pokemon["name"]
        for type_info in pokemon.get("types", []):
            type_name = type_info["type"]["name"]
            type_index[type_name].append(name)

    return dict(type_index)


# ---------------------------------------------------------------------------
# Index par statut (légendaire, mythique, bébé)
# ---------------------------------------------------------------------------

def build_status_index(pokemon_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Construit un index des Pokémon par statut (légendaire, mythique, bébé)."""
    status_index: Dict[str, List[str]] = {
        "legendary": [],
        "mythical": [],
        "baby": []
    }

    for pokemon in pokemon_data:
        name = pokemon["name"]
        species_info = pokemon.get("species_info", {})

        if species_info.get("is_legendary"):
            status_index["legendary"].append(name)
        if species_info.get("is_mythical"):
            status_index["mythical"].append(name)
        if species_info.get("is_baby"):
            status_index["baby"].append(name)

    return status_index


# ---------------------------------------------------------------------------
# Index par habitat
# ---------------------------------------------------------------------------

def build_habitat_index(pokemon_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Construit un index des Pokémon par habitat."""
    habitat_index: Dict[str, List[str]] = defaultdict(list)

    for pokemon in pokemon_data:
        name = pokemon["name"]
        species_info = pokemon.get("species_info", {})

        habitat = species_info.get("habitat")
        if habitat and isinstance(habitat, dict):
            habitat_name = habitat.get("name", "")
            if habitat_name:
                habitat_index[habitat_name].append(name)

    return dict(habitat_index)


# ---------------------------------------------------------------------------
# Index par couleur
# ---------------------------------------------------------------------------

def build_color_index(pokemon_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Construit un index des Pokémon par couleur."""
    color_index: Dict[str, List[str]] = defaultdict(list)

    for pokemon in pokemon_data:
        name = pokemon["name"]
        species_info = pokemon.get("species_info", {})

        color = species_info.get("color")
        if color and isinstance(color, dict):
            color_name = color.get("name", "")
            if color_name:
                color_index[color_name].append(name)

    return dict(color_index)


# ---------------------------------------------------------------------------
# Outil de sauvegarde générique
# ---------------------------------------------------------------------------

def save_index(index: Dict[str, Any], filename: str):
    """Sauvegarde un index dans un fichier JSON sous data/indexes."""
    output_dir = "data/indexes"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    print("Chargement des données Pokémon…")
    pokemon_data = load_pokemon_data()

    print("Construction des index (hors évolutions)…")

    type_index = build_type_index(pokemon_data)
    status_index = build_status_index(pokemon_data)
    habitat_index = build_habitat_index(pokemon_data)
    color_index = build_color_index(pokemon_data)

    print("Sauvegarde des index…")
    save_index(type_index, "type_index.json")
    save_index(status_index, "status_index.json")
    save_index(habitat_index, "habitat_index.json")
    save_index(color_index, "color_index.json")

    # Statistiques rapides
    print("\nStatistiques des index :")
    print(f"Nombre de types différents : {len(type_index)}")
    print(f"Nombre de Pokémon légendaires : {len(status_index['legendary'])}")
    print(f"Nombre de Pokémon mythiques : {len(status_index['mythical'])}")
    print(f"Nombre de bébés Pokémon : {len(status_index['baby'])}")
    print(f"Nombre d'habitats différents : {len(habitat_index)}")
    print(f"Nombre de couleurs différentes : {len(color_index)}")


if __name__ == "__main__":
    main()
