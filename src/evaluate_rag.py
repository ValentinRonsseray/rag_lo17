"""
script d'évaluation rag pokémon
"""

import json
import asyncio
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import sys
import tempfile
import shutil
import atexit
import re

# ajout du répertoire racine au path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator
from src.format_pokeapi_data import create_pokemon_documents

# questions de test
TEST_QUESTIONS = [
    {
        "question": "Quels sont les Pokémon de type fire ?",
        "reference": "arcanine-hisui, arcanine, charizard-gmax, charizard-mega-x, charizard-mega-y, charizard, charmander, charmeleon, flareon, growlithe-hisui, growlithe, magmar, marowak-alola, moltres, ninetales, ponyta, rapidash, vulpix",
        "type": "semantic",
    },
    {
        "question": "Décris-moi Arcanin",
        "reference": "Arcanin est un Pokémon de type Feu qui a les capacités intimidate, flash-fire et justified. Ses statistiques sont : PV 90, Attaque 110, Défense 80, Attaque Spéciale 100, Défense Spéciale 80 et Vitesse 95.",
        "type": "semantic",
    },
    {
        "question": "Liste les Pokémon légendaires",
        "reference": "articuno-galar, articuno, mewtwo-mega-x, mewtwo-mega-y, mewtwo, moltres-galar, moltres, zapdos-galar, zapdos",
        "type": "semantic",
    },
    {
        "question": "Quelles sont les statistiques de base de Pikachu ?",
        "reference": "PV 35, Attaque 55, Défense 40, Attaque Spéciale 50, Défense Spéciale 50, Vitesse 90.",
        "type": "semantic",
    },
    {
        "question": "Quels sont les Pokémon mythiques ?",
        "reference": "mew",
        "type": "semantic",
    },
    {
        "question": "Quels Pokémon vivent dans les caves ?",
        "reference": "diglett-alola, diglett, dugtrio-alola, dugtrio, gastly, gengar-gmax, gengar-mega, gengar, golbat, haunter, onix, zubat",
        "type": "semantic",
    }
]


async def evaluate_response(
    evaluator: RAGEvaluator, result: Dict[str, Any], test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """évalue une réponse"""
    scores = await evaluator.evaluate_response(
        result["answer"], test_case["reference"], result["context"]
    )

    return {
        "question": test_case["question"],
        "expected_type": test_case["type"],
        "actual_type": result.get("search_type", "semantic"),
        "prediction": result["answer"],
        "reference": test_case["reference"],
        **scores,
    }


def save_results(results_df: pd.DataFrame, output_dir: Path):
    """sauvegarde les résultats"""
    final_dir = Path("evaluation_results")
    try:
        # supprime le dossier existant
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)

        # crée le dossier
        final_dir.mkdir(exist_ok=True)

        # copie les fichiers
        for file in output_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, final_dir / file.name)
    except Exception as e:
        print(f"erreur de sauvegarde : {e}")


async def run_evaluation():
    """lance l'évaluation"""
    print("initialisation...")
    rag_system = RAGSystem()
    evaluator = RAGEvaluator()

    # charge les documents
    print("chargement des documents...")
    documents = create_pokemon_documents()
    rag_system.embed_documents(documents)

    # prépare les résultats
    results = []
    output_dir = None

    try:
        # crée le dossier temporaire
        output_dir = Path(tempfile.mkdtemp(prefix="eval_results_"))

        # évalue chaque question
        print("\ndébut de l'évaluation...")
        for i, test_case in enumerate(TEST_QUESTIONS, 1):
            print(f"\ntest {i}/{len(TEST_QUESTIONS)}: {test_case['question']}")

            # obtient la réponse
            result = rag_system.query(test_case["question"])
        

            # évalue
            result_data = await evaluate_response(evaluator, result, test_case)
            results.append(result_data)

            # affiche les résultats
            print(f"type de recherche: {result.get('search_type', 'semantic')}")
            print(f"correspondance exacte: {result_data['exact_match']:.2f}")
            print(f"score f1: {result_data['f1_score']:.2f}")
            print(f"fidélité: {result_data['faithfulness']:.2f}")

        # crée le dataframe
        results_df = pd.DataFrame(results)

        # sauvegarde les résultats
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        # génère les graphiques
        await evaluator.plot_results(results_df, output_dir)

        # sauvegarde dans le dossier final
        save_results(results_df, output_dir)

        # analyse des résultats
        print("\nanalyse des résultats:")
        print("\nmoyennes par type:")
        print(
            results_df.groupby("actual_type")[
                ["exact_match", "f1_score", "faithfulness"]
            ].mean()
        )

        print("\nmoyennes globales:")
        print(results_df[["exact_match", "f1_score", "faithfulness"]].mean())

        # analyse des erreurs
        print("\nanalyse des erreurs:")
        low_faithfulness = results_df[results_df["faithfulness"] < 0.7]
        if not low_faithfulness.empty:
            print("\nquestions avec faible fidélité:")
            for _, row in low_faithfulness.iterrows():
                print(f"\nquestion: {row['question']}")
                print(f"prédiction: {row['prediction']}")
                print(f"référence: {row['reference']}")
                print(f"score de fidélité: {row['faithfulness']:.2f}")

    finally:
        # nettoie le dossier temporaire
        if output_dir and output_dir.exists():
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as e:
                print(f"erreur de nettoyage : {e}")


def cleanup():
    """nettoie les ressources"""
    try:
        # nettoie le dossier temporaire
        temp_dir = Path("chroma_db")
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"erreur de nettoyage : {e}")


if __name__ == "__main__":
    # enregistre la fonction de nettoyage
    atexit.register(cleanup)

    # lance l'évaluation
    asyncio.run(run_evaluation())
