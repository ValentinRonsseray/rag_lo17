"""
script de construction des index pokémon
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict


def load_pokemon_data() -> List[Dict[str, Any]]:
    """charge les données pokémon"""
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
    """construit l'index par type"""
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
    """construit l'index par statut"""
    status_index: Dict[str, List[str]] = {"legendary": [], "mythical": [], "baby": []}

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
    """construit l'index par habitat"""
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
    """construit l'index par couleur"""
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
    """sauvegarde un index"""
    output_dir = "data/indexes"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def main():
    print("chargement des données…")
    pokemon_data = load_pokemon_data()

    print("construction des index…")

    type_index = build_type_index(pokemon_data)
    status_index = build_status_index(pokemon_data)
    habitat_index = build_habitat_index(pokemon_data)
    color_index = build_color_index(pokemon_data)

    print("sauvegarde des index…")
    save_index(type_index, "type_index.json")
    save_index(status_index, "status_index.json")
    save_index(habitat_index, "habitat_index.json")
    save_index(color_index, "color_index.json")

    # stats
    print("\nstats des index :")
    print(f"nombre de types : {len(type_index)}")
    print(f"nombre de légendaires : {len(status_index['legendary'])}")
    print(f"nombre de mythiques : {len(status_index['mythical'])}")
    print(f"nombre de bébés : {len(status_index['baby'])}")
    print(f"nombre d'habitats : {len(habitat_index)}")
    print(f"nombre de couleurs : {len(color_index)}")


if __name__ == "__main__":
    main()
