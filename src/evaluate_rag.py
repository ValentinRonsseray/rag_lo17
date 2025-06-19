"""
script d'évaluation rag pokémon
"""

import asyncio
import atexit
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ajout du répertoire racine au path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator
from src.format_pokeapi_data import create_pokemon_documents


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Charge un fichier JSON contenant les questions de test."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


async def run_evaluation(dataset_path: Path | None = None) -> None:
    """Lance l'évaluation RAG sur un jeu de questions."""
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

        if dataset_path is None:
            raise ValueError("Chemin du jeu de questions non fourni")

        test_questions = load_questions(dataset_path)

        # évalue chaque question
        print("\ndébut de l'évaluation...")
        for i, test_case in enumerate(test_questions, 1):
            print(f"\ntest {i}/{len(test_questions)}: {test_case['question']}")

            # obtient la réponse
            result = rag_system.query(test_case["question"])
        

            # évalue
            result_data = await evaluate_response(evaluator, result, test_case)
            results.append(result_data)

            # affiche les résultats
            print(f"type de recherche: {result.get('search_type', 'semantic')}")
            print(f"f1 réponse: {result_data['answer_f1']:.2f}")
            print(f"similarité: {result_data['answer_similarity']:.2f}")
            print(f"precision contexte: {result_data['context_precision']:.2f}")
            print(f"rappel contexte: {result_data['context_recall']:.2f}")
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
                [
                    "answer_f1",
                    "answer_similarity",
                    "context_precision",
                    "context_recall",
                    "faithfulness",
                ]
            ].mean()
        )

        print("\nmoyennes globales:")
        print(
            results_df[
                [
                    "answer_f1",
                    "answer_similarity",
                    "context_precision",
                    "context_recall",
                    "faithfulness",
                ]
            ].mean()
        )

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

    # chemin du jeu de questions
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # lance l'évaluation
    asyncio.run(run_evaluation(dataset))
