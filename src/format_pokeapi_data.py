"""
script pour formater les données pokeapi pour le système rag.
"""

import json
import os
from typing import List, Dict, Any
from langchain.docstore.document import Document
from src.pokepedia_data import PokepediaData


def load_pokemon_data() -> List[Dict[str, Any]]:
    """charge les données pokémon depuis les fichiers json."""
    data_dir = "data/pokeapi"
    pokemon_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                pokemon_data.append(json.load(f))

    return pokemon_data


def format_pokemon_document(
    pokemon: Dict[str, Any], pokepedia: PokepediaData
) -> Document:
    """formate les données d'un pokémon en document pour le rag."""
    # informations de base
    name = pokemon.get("name", "")
    base_form = pokemon.get("base_form", name)

    # types
    types = [t["type"]["name"] for t in pokemon.get("types", [])]
    types_str = " et ".join(types)

    # statistiques
    stats = {
        stat["stat"]["name"]: stat["base_stat"] for stat in pokemon.get("stats", [])
    }

    # capacités
    abilities = [a["ability"]["name"] for a in pokemon.get("abilities", [])]
    abilities_str = ", ".join(abilities)

    # informations sur l'espèce
    species_info = pokemon.get("species_info", {}) or {}
    flavor_text = ""
    names = {}
    genera = {}

    # récupération de la description en français
    for entry in species_info.get("flavor_text_entries", []):
        if entry.get("language", {}).get("name") == "fr":
            flavor_text = entry.get("flavor_text", "").replace("\n", " ")
            break

    # récupération des noms dans différentes langues
    for name_entry in species_info.get("names", []):
        lang = name_entry.get("language", {}).get("name")
        if lang in ["en", "fr", "ja"]:
            names[lang] = name_entry.get("name", "")

    # récupération des genres dans différentes langues
    for genus_entry in species_info.get("genera", []):
        lang = genus_entry.get("language", {}).get("name")
        if lang in ["en", "fr", "ja"]:
            genera[lang] = genus_entry.get("genus", "")

    # construction du texte
    text = f"le pokémon {name}"
    if name != base_form:
        text += f" (forme de {base_form})"

    text += f" est de type {types_str}. "
    text += f"il possède les capacités suivantes : {abilities_str}. "

    if stats:
        text += f"ses statistiques de base sont : "
        stats_text = []
        for stat_name, value in stats.items():
            stat_name_fr = {
                "hp": "pv",
                "attack": "attaque",
                "defense": "défense",
                "special-attack": "attaque spéciale",
                "special-defense": "défense spéciale",
                "speed": "vitesse",
            }.get(stat_name, stat_name)
            stats_text.append(f"{stat_name_fr}: {value}")
        text += ", ".join(stats_text) + ". "

    # ajout des informations poképédia
    pokepedia_info = pokemon.get("pokepedia", {})
    if pokepedia_info:
        if pokepedia_info.get("description"):
            text += f"\n\n{pokepedia_info['description']}"

        if pokepedia_info.get("biology"):
            text += f"\n\nbiologie : {pokepedia_info['biology']}"

        if pokepedia_info.get("behavior"):
            text += f"\n\ncomportement : {pokepedia_info['behavior']}"

        if pokepedia_info.get("habitat"):
            text += f"\n\nhabitat : {pokepedia_info['habitat']}"

        if pokepedia_info.get("evolution"):
            text += f"\n\névolution : {pokepedia_info['evolution']}"

        if pokepedia_info.get("mythology"):
            text += f"\n\nmythologie : {pokepedia_info['mythology']}"

        if pokepedia_info.get("trivia"):
            text += "\n\nfaits divers :"
            for trivia in pokepedia_info["trivia"]:
                text += f"\n- {trivia}"

    elif flavor_text:
        text += f"\n\ndescription : {flavor_text}"

    # métadonnées
    metadata = {
        "source": "pokeapi",
        "name": name,
        "base_form": base_form,
        "types": ", ".join(types),
        "abilities": ", ".join(abilities),
        "stats": json.dumps(stats),
        "names": json.dumps(names),
        "genera": json.dumps(genera),
        "is_legendary": species_info.get("is_legendary", False),
        "is_mythical": species_info.get("is_mythical", False),
        "is_baby": species_info.get("is_baby", False),
        "color": (
            species_info.get("color", {}).get("name", "")
            if species_info.get("color")
            else ""
        ),
        "habitat": (
            species_info.get("habitat", {}).get("name", "")
            if species_info.get("habitat")
            else ""
        ),
        "has_pokepedia": bool(pokepedia_info),
    }

    return Document(page_content=text, metadata=metadata)


def create_pokemon_documents() -> List[Document]:
    """crée les documents pour tous les pokémon."""
    pokemon_data = load_pokemon_data()
    pokepedia = PokepediaData()

    # enrichissement des données avec poképédia
    enriched_data = [
        pokepedia.enrich_pokemon_document(pokemon) for pokemon in pokemon_data
    ]

    return [format_pokemon_document(pokemon, pokepedia) for pokemon in enriched_data]


if __name__ == "__main__":
    documents = create_pokemon_documents()
    print(f"nombre de documents créés : {len(documents)}")
    print("\nexemple de document :")
    print(documents[0].page_content)
    print("\nmétadonnées :")
    print(documents[0].metadata)
