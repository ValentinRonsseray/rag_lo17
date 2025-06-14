"""
Script d'évaluation du système RAG Pokémon.
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

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator
from src.format_pokeapi_data import create_pokemon_documents

# Questions de test avec leurs réponses de référence
# Nouvelles questions de test couvrant différents types de recherche
TEST_QUESTIONS = [
    {
        "question": "Quels sont les Pokémon de type feu ?",
        "reference": "Les Pokémon de type feu sont : Charmander, Charmeleon, Charizard, Vulpix, Ninetales, Growlithe et Arcanine",
        "type": "exact",
    },
    {
        "question": "Décris-moi Arcanin",
        "reference": "Arcanin est un Pokémon majestueux de type feu. Son port altier et son attitude fière ont depuis longtemps conquis le cœur des hommes. Il possède un pelage principalement orange avec des marques noires et blanches, une crinière imposante et une queue touffue. Arcanin est connu pour sa loyauté et son courage. Il est extrêmement fidèle à son dresseur et protégera son territoire avec ferveur. Malgré son apparence imposante, il est doux avec les personnes qu'il apprécie.",
        "type": "semantic",
    },
    {
        "question": "Liste les Pokémon légendaires",
        "reference": "Les Pokémon légendaires sont : Articuno, Zapdos, Moltres, Mewtwo et Lugia",
        "type": "exact",
    },
    {
        "question": "Quelles sont les statistiques de base de Pikachu ?",
        "reference": "Les statistiques de base de Pikachu sont : PV 35, Attaque 55, Défense 40, Attaque Spéciale 50, Défense Spéciale 50 et Vitesse 90",
        "type": "semantic",
    },
    {
        "question": "Quels sont les Pokémon mythiques ?",
        "reference": "Les Pokémon mythiques sont : Mew, Celebi, Jirachi, Deoxys et Darkrai",
        "type": "exact",
    },
    {
        "question": "Quels Pokémon vivent dans les forêts ?",
        "reference": "Les Pokémon qui vivent dans les forêts comprennent : Caterpie, Pikachu, Oddish et Paras",
        "type": "exact",
    },
    {
        "question": "Quels sont les Pokémon de couleur rouge ?",
        "reference": "Les Pokémon de couleur rouge sont : Charmander, Vulpix, Paras et Magmar",
        "type": "exact",
    },
    {
        "question": "Quelles sont les évolutions d'Évoli ?",
        "reference": "Les évolutions d'Évoli sont : Vaporeon, Jolteon, Flareon, Espeon, Umbreon, Leafeon, Glaceon et Sylveon",
        "type": "semantic",
    },
]

async def evaluate_response(evaluator: RAGEvaluator, result: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Évalue une réponse individuelle."""
    scores = await evaluator.evaluate_response(
        result["answer"],
        test_case["reference"],
        result["context"]
    )
    
    return {
        "question": test_case["question"],
        "expected_type": test_case["type"],
        "actual_type": result.get("search_type", "semantic"),
        "prediction": result["answer"],
        "reference": test_case["reference"],
        **scores
    }

def save_results(results_df: pd.DataFrame, output_dir: Path):
    """Sauvegarde les résultats dans le dossier final."""
    final_dir = Path("evaluation_results")
    try:
        # Supprimer le dossier final s'il existe
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        
        # Créer le dossier final
        final_dir.mkdir(exist_ok=True)
        
        # Copier les fichiers un par un
        for file in output_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, final_dir / file.name)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats : {e}")

async def run_evaluation():
    """Exécute l'évaluation complète du système RAG."""
    print("Initialisation du système RAG...")
    rag_system = RAGSystem()
    evaluator = RAGEvaluator()
    
    # Chargement des documents
    print("Chargement des documents Pokémon...")
    documents = create_pokemon_documents()
    rag_system.embed_documents(documents)
    
    # Préparation des résultats
    results = []
    output_dir = None
    
    try:
        # Créer un dossier temporaire pour les résultats
        output_dir = Path(tempfile.mkdtemp(prefix="eval_results_"))
        
        # Évaluation de chaque question
        print("\nDébut de l'évaluation...")
        for i, test_case in enumerate(TEST_QUESTIONS, 1):
            print(f"\nTest {i}/{len(TEST_QUESTIONS)}: {test_case['question']}")
            
            # Obtention de la réponse
            result = rag_system.query(test_case['question'])
            
            # Évaluation
            result_data = await evaluate_response(evaluator, result, test_case)
            results.append(result_data)
            
            # Affichage des résultats
            print(f"Type de recherche: {result.get('search_type', 'semantic')}")
            print(f"Correspondance exacte: {result_data['exact_match']:.2f}")
            print(f"Score F1: {result_data['f1_score']:.2f}")
            print(f"Fidélité: {result_data['faithfulness']:.2f}")
        
        # Création du DataFrame des résultats
        results_df = pd.DataFrame(results)
        
        # Sauvegarde des résultats
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
        
        # Génération des graphiques
        await evaluator.plot_results(results_df, output_dir)
        
        # Sauvegarde des résultats dans le dossier final
        save_results(results_df, output_dir)
        
        # Analyse des résultats
        print("\nAnalyse des résultats:")
        print("\nMoyennes par type de recherche:")
        print(results_df.groupby("actual_type")[["exact_match", "f1_score", "faithfulness"]].mean())
        
        print("\nMoyennes globales:")
        print(results_df[["exact_match", "f1_score", "faithfulness"]].mean())
        
        # Analyse des erreurs
        print("\nAnalyse des erreurs:")
        low_faithfulness = results_df[results_df["faithfulness"] < 0.7]
        if not low_faithfulness.empty:
            print("\nQuestions avec faible fidélité:")
            for _, row in low_faithfulness.iterrows():
                print(f"\nQuestion: {row['question']}")
                print(f"Prédiction: {row['prediction']}")
                print(f"Référence: {row['reference']}")
                print(f"Score de fidélité: {row['faithfulness']:.2f}")
    
    finally:
        # Nettoyage du dossier temporaire
        if output_dir and output_dir.exists():
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as e:
                print(f"Erreur lors du nettoyage : {e}")

def cleanup():
    """Fonction de nettoyage appelée à la sortie."""
    try:
        # Nettoyer le dossier temporaire si nécessaire
        temp_dir = Path("chroma_db")
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Erreur lors du nettoyage : {e}")

if __name__ == "__main__":
    # Enregistrer la fonction de nettoyage
    atexit.register(cleanup)
    
    # Exécuter l'évaluation
    asyncio.run(run_evaluation()) 