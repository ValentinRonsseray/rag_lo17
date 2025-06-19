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
    output_dir = "evaluation_results"

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
        print("\n" + "="*60)
        print("ANALYSE DÉTAILLÉE DES RÉSULTATS")
        print("="*60)
        
        # Préparer le contenu pour le fichier texte
        report_content = []
        report_content.append("="*60)
        report_content.append("RAPPORT D'ÉVALUATION RAG POKÉMON")
        report_content.append("="*60)
        report_content.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Nombre total de questions: {len(results_df)}")
        report_content.append("")
        
        # Statistiques globales
        print("\nSTATISTIQUES GLOBALES:")
        print("-" * 40)
        report_content.append("STATISTIQUES GLOBALES:")
        report_content.append("-" * 40)
        
        global_stats = results_df[
            [
                "answer_f1",
                "answer_similarity", 
                "context_precision",
                "context_recall",
                "faithfulness",
            ]
        ].agg(['mean', 'std', 'min', 'max', 'median']).round(3)
        print(global_stats)
        report_content.append(str(global_stats))
        report_content.append("")
        
        # Analyse par type de question
        print("\nANALYSE PAR TYPE DE QUESTION:")
        print("-" * 40)
        report_content.append("ANALYSE PAR TYPE DE QUESTION:")
        report_content.append("-" * 40)
        
        type_stats = results_df.groupby("expected_type")[
            [
                "answer_f1",
                "answer_similarity",
                "context_precision", 
                "context_recall",
                "faithfulness",
            ]
        ].agg(['mean', 'count']).round(3)
        print(type_stats)
        report_content.append(str(type_stats))
        report_content.append("")
        
        # Analyse par type de recherche
        print("\nANALYSE PAR TYPE DE RECHERCHE:")
        print("-" * 40)
        report_content.append("ANALYSE PAR TYPE DE RECHERCHE:")
        report_content.append("-" * 40)
        
        search_stats = results_df.groupby("actual_type")[
            [
                "answer_f1",
                "answer_similarity",
                "context_precision",
                "context_recall", 
                "faithfulness",
            ]
        ].agg(['mean', 'count']).round(3)
        print(search_stats)
        report_content.append(str(search_stats))
        report_content.append("")
        
        # Distribution des scores
        print("\nDISTRIBUTION DES SCORES:")
        print("-" * 40)
        report_content.append("DISTRIBUTION DES SCORES:")
        report_content.append("-" * 40)
        
        for metric in ["answer_f1", "answer_similarity", "context_precision", "context_recall", "faithfulness"]:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")
            
            excellent = len(results_df[results_df[metric] >= 0.9])
            good = len(results_df[(results_df[metric] >= 0.7) & (results_df[metric] < 0.9)])
            medium = len(results_df[(results_df[metric] >= 0.5) & (results_df[metric] < 0.7)])
            poor = len(results_df[results_df[metric] < 0.5])
            total = len(results_df)
            
            print(f"  Excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)")
            print(f"  Bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)")
            print(f"  Moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)")
            print(f"  Faible (<0.5): {poor} questions ({poor/total*100:.1f}%)")
            
            report_content.append(f"  Excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)")
            report_content.append(f"  Bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)")
            report_content.append(f"  Moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)")
            report_content.append(f"  Faible (<0.5): {poor} questions ({poor/total*100:.1f}%)")
        
        # Corrélations entre métriques
        print("\nCORRÉLATIONS ENTRE MÉTRIQUES:")
        print("-" * 40)
        report_content.append("\nCORRÉLATIONS ENTRE MÉTRIQUES:")
        report_content.append("-" * 40)
        
        correlation_matrix = results_df[
            ["answer_f1", "answer_similarity", "context_precision", "context_recall", "faithfulness"]
        ].corr().round(3)
        print(correlation_matrix)
        report_content.append(str(correlation_matrix))
        report_content.append("")
        
        # Questions avec les meilleurs scores
        print("\nTOP 3 QUESTIONS PAR MÉTRIQUE:")
        print("-" * 40)
        report_content.append("TOP 3 QUESTIONS PAR MÉTRIQUE:")
        report_content.append("-" * 40)
        
        for metric in ["answer_f1", "answer_similarity", "faithfulness"]:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")
            
            top_3 = results_df.nlargest(3, metric)[["question", metric]]
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                line = f"  {i}. {row['question'][:60]}... (score: {row[metric]:.3f})"
                print(line)
                report_content.append(line)
        
        # Questions avec les plus mauvais scores
        print("\nQUESTIONS AVEC LES PLUS MAUVAIS SCORES:")
        print("-" * 40)
        report_content.append("\nQUESTIONS AVEC LES PLUS MAUVAIS SCORES:")
        report_content.append("-" * 40)
        
        for metric in ["answer_f1", "faithfulness"]:
            print(f"\n{metric.upper()} (plus bas):")
            report_content.append(f"\n{metric.upper()} (plus bas):")
            
            worst_3 = results_df.nsmallest(3, metric)[["question", metric]]
            for i, (_, row) in enumerate(worst_3.iterrows(), 1):
                line = f"  {i}. {row['question'][:60]}... (score: {row[metric]:.3f})"
                print(line)
                report_content.append(line)
        
        # Analyse des erreurs détaillée
        print("\nANALYSE DÉTAILLÉE DES ERREURS:")
        print("-" * 40)
        report_content.append("\nANALYSE DÉTAILLÉE DES ERREURS:")
        report_content.append("-" * 40)
        
        # Questions avec faible fidélité
        low_faithfulness = results_df[results_df["faithfulness"] < 0.7]
        if not low_faithfulness.empty:
            print(f"\nQuestions avec faible fidélité (<0.7): {len(low_faithfulness)}")
            print(f"Moyenne fidélité pour ces questions: {low_faithfulness['faithfulness'].mean():.3f}")
            report_content.append(f"\nQuestions avec faible fidélité (<0.7): {len(low_faithfulness)}")
            report_content.append(f"Moyenne fidélité pour ces questions: {low_faithfulness['faithfulness'].mean():.3f}")
            
            for _, row in low_faithfulness.iterrows():
                print(f"\n  Question: {row['question']}")
                print(f"  Prédiction: {row['prediction'][:100]}...")
                print(f"  Référence: {row['reference'][:100]}...")
                print(f"  Score fidélité: {row['faithfulness']:.3f}")
                
                report_content.append(f"\n  Question: {row['question']}")
                report_content.append(f"  Prédiction: {row['prediction'][:100]}...")
                report_content.append(f"  Référence: {row['reference'][:100]}...")
                report_content.append(f"  Score fidélité: {row['faithfulness']:.3f}")
        else:
            print("Toutes les questions ont une bonne fidélité (≥0.7)")
            report_content.append("Toutes les questions ont une bonne fidélité (≥0.7)")
        
        # Questions avec faible F1
        low_f1 = results_df[results_df["answer_f1"] < 0.5]
        if not low_f1.empty:
            print(f"\nQuestions avec faible F1 (<0.5): {len(low_f1)}")
            print(f"Moyenne F1 pour ces questions: {low_f1['answer_f1'].mean():.3f}")
            report_content.append(f"\nQuestions avec faible F1 (<0.5): {len(low_f1)}")
            report_content.append(f"Moyenne F1 pour ces questions: {low_f1['answer_f1'].mean():.3f}")
        
        # Résumé des performances
        print("\nRÉSUMÉ DES PERFORMANCES:")
        print("-" * 40)
        report_content.append("\nRÉSUMÉ DES PERFORMANCES:")
        report_content.append("-" * 40)
        
        summary_lines = [
            f"Nombre total de questions: {len(results_df)}",
            f"Score F1 moyen: {results_df['answer_f1'].mean():.3f} ± {results_df['answer_f1'].std():.3f}",
            f"Similarité moyenne: {results_df['answer_similarity'].mean():.3f} ± {results_df['answer_similarity'].std():.3f}",
            f"Fidélité moyenne: {results_df['faithfulness'].mean():.3f} ± {results_df['faithfulness'].std():.3f}",
            f"Précision contexte moyenne: {results_df['context_precision'].mean():.3f} ± {results_df['context_precision'].std():.3f}",
            f"Rappel contexte moyen: {results_df['context_recall'].mean():.3f} ± {results_df['context_recall'].std():.3f}"
        ]
        
        for line in summary_lines:
            print(line)
            report_content.append(line)
        
        # Sauvegarder le rapport dans un fichier texte
        report_filename = "evaluation_report.txt"
        report_path = Path("evaluation_results") / report_filename
        
        # Créer le dossier s'il n'existe pas
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        
        print(f"\nRapport détaillé sauvegardé: {report_path}")
        report_content.append(f"\nRapport détaillé sauvegardé: {report_path}")

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


def create_sample_questions():
    """Crée un fichier de questions d'exemple pour les tests."""
    import json
    
    sample_questions = [
        {
            "question": "Quelles sont les statistiques de base de Pikachu ?",
            "reference": "Pikachu a 35 points de vie, 55 d'attaque, 40 de défense, 50 d'attaque spéciale, 50 de défense spéciale et 90 de vitesse.",
            "type": "statistics"
        },
        {
            "question": "Décris le comportement de Charizard",
            "reference": "Charizard est un Pokémon fier et courageux qui aime les défis. Il est très loyal envers son dresseur et protège son territoire avec ferveur.",
            "type": "description"
        },
        {
            "question": "Quels sont les Pokémon de type feu ?",
            "reference": "Les Pokémon de type feu incluent Salamèche, Reptincel, Dracaufeu, Caninos, Arcanin, Ponyta, Galopa, et d'autres.",
            "type": "categorization"
        },
        {
            "question": "Quels sont les Pokémon légendaires ?",
            "reference": "Les Pokémon légendaires incluent Articuno, Zapdos, Moltres, Mewtwo, Mew, et d'autres.",
            "type": "categorization"
        },
        {
            "question": "Parle-moi de l'habitat de Bulbizarre",
            "reference": "Bulbizarre vit principalement dans les forêts et les prairies. Il préfère les endroits ensoleillés où il peut absorber la lumière du soleil.",
            "type": "description"
        }
    ]
    
    # Crée le dossier data s'il n'existe pas
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sauvegarde le fichier
    with open(data_dir / "test_questions.json", "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, ensure_ascii=False, indent=2)
    
    print(f"Fichier de questions d'exemple créé: {data_dir / 'test_questions.json'}")


if __name__ == "__main__":
    # enregistre la fonction de nettoyage
    atexit.register(cleanup)

    # chemin du jeu de questions
    if len(sys.argv) > 1:
        dataset = Path(sys.argv[1])
    else:
        # Utilise un fichier par défaut s'il existe
        default_dataset = Path("data/test_questions.json")
        if default_dataset.exists():
            dataset = default_dataset
            print(f"Utilisation du fichier de questions par défaut: {dataset}")
        else:
            print("Aucun fichier de questions fourni et aucun fichier par défaut trouvé.")
            print("Création d'un fichier de questions d'exemple...")
            create_sample_questions()
            dataset = default_dataset

    # lance l'évaluation
    asyncio.run(run_evaluation(dataset))
