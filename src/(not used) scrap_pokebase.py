"""
download_pokemon_data.py

Script pour télécharger toutes les données Pokémon depuis PokéAPI
et les sauvegarder sous forme JSON et CSV.

Usage :
    python download_pokemon_data.py [--out ./data] [--rate 0.25]

Options :
    --out   Dossier de sortie où seront créés les fichiers (défaut : ./data)
    --rate  Délai (en secondes) entre deux requêtes pour respecter la politique de fair-use (défaut : 0.25s)

Dépendances :
    pip install requests pandas tqdm
"""

import argparse
import os
import time
import json
from pathlib import Path

import requests
from tqdm import tqdm
import pandas as pd

BASE_URL = "https://pokeapi.co/api/v2"

def fetch(endpoint: str):
    """Effectue une requête GET et retourne le JSON."""
    resp = requests.get(endpoint, timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_all_pokemon(rate: float):
    """Récupère la liste complète puis les détails de chaque Pokémon."""
    print("Récupération de la liste des Pokémon…")
    payload = fetch(f"{BASE_URL}/pokemon?limit=2000")
    pokemon_urls = [p["url"] for p in payload["results"]]

    all_data = []
    for url in tqdm(pokemon_urls, desc="Téléchargement des détails"):
        all_data.append(fetch(url))
        time.sleep(rate)  # rate-limiting
    return all_data

def normalize_for_csv(raw):
    """Aplati les champs principaux pour un DataFrame."""
    records = []
    for p in raw:
        stats = {s["stat"]["name"]: s["base_stat"] for s in p["stats"]}
        types = ",".join(t["type"]["name"] for t in sorted(p["types"], key=lambda t: t["slot"]))
        abilities = ",".join(a["ability"]["name"] for a in p["abilities"])
        records.append(
            {
                "id": p["id"],
                "name": p["name"],
                "types": types,
                "abilities": abilities,
                **stats,  # hp, attack, defense, special-attack, special-defense, speed
                "weight": p["weight"],
                "height": p["height"],
            }
        )
    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser(description="Télécharge toutes les données Pokémon depuis PokéAPI.")
    parser.add_argument("--out", default="./data", help="Dossier de sortie.")
    parser.add_argument("--rate", type=float, default=0.25, help="Pause (secondes) entre requêtes.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_data = get_all_pokemon(args.rate)

    # Sauvegarde JSON
    json_file = out_dir / "pokemon_raw.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"Données brutes sauvegardées dans {json_file}")

    # Sauvegarde CSV
    df = normalize_for_csv(raw_data)
    csv_file = out_dir / "pokemon_basic_stats.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV sauvegardé dans {csv_file}")

if __name__ == "__main__":
    main()
