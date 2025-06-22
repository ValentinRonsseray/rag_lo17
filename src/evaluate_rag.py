"""
python evaluate_rag.py pour lancer l'évaluation en mode normal
python evaluate_rag.py --engaged pour lancer l'évaluation en mode engagé
python evaluate_rag.py --help pour afficher l'aide et les options
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
from src.evaluation import RAGEvaluator, evaluate_with_metrics
from src.format_pokeapi_data import create_pokemon_documents


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """charge un fichier json contenant les questions de test."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate_response(
    evaluator: RAGEvaluator, result: Dict[str, Any], test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """évalue une réponse avec ragas"""
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


def save_results(results_df: pd.DataFrame, output_dir: Path, engaged_mode: bool = False):
    """sauvegarde les résultats"""
    final_dir = Path("evaluation_results")
    try:
        # supprime le dossier existant
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)

        # crée le dossier
        final_dir.mkdir(exist_ok=True)

        # copie les fichiers avec suffixe si mode engagé
        for file in output_dir.glob("*"):
            if file.is_file():
                if engaged_mode and not file.name.startswith("evaluation_report"):
                    # ajoute le suffixe _engaged aux fichiers de résultats
                    new_name = file.stem + "_engaged" + file.suffix
                    shutil.copy2(file, final_dir / new_name)
                else:
                    shutil.copy2(file, final_dir / file.name)
    except Exception as e:
        print(f"erreur de sauvegarde : {e}")


async def run_evaluation_in_batches(dataset_path: Path | None = None, batch_size: int = 10, engaged_mode: bool = False) -> None:
    """lance l'évaluation rag par lots pour éviter les limites de quota."""
    print("initialisation...")
    print(f"mode engagé: {'activé' if engaged_mode else 'désactivé'}")
    
    rag_system = RAGSystem(engaged_mode=engaged_mode)
    evaluator = RAGEvaluator()

    # charge les documents
    print("chargement des documents...")
    documents = create_pokemon_documents()
    rag_system.embed_documents(documents)

    # charge les questions
    if dataset_path is None:
        raise ValueError("chemin du jeu de questions non fourni")

    test_questions = load_questions(dataset_path)
    print(f"total questions à évaluer: {len(test_questions)}")

    # découpe en lots
    batches = [test_questions[i:i + batch_size] for i in range(0, len(test_questions), batch_size)]
    print(f"découpage en {len(batches)} lots de {batch_size} questions")

    # prépare les résultats
    all_results = []
    output_dir = "evaluation_results"

    try:
        # crée le dossier temporaire
        output_dir = Path(tempfile.mkdtemp(prefix="eval_results_"))

        # traite chaque lot
        for batch_idx, batch in enumerate(batches, 1):
            print(f"\n{'='*60}")
            print(f"LOT {batch_idx}/{len(batches)} ({len(batch)} questions)")
            print(f"{'='*60}")

            batch_results = []

            # évalue chaque question du lot
            for i, test_case in enumerate(batch, 1):
                global_idx = (batch_idx - 1) * batch_size + i
                print(f"\ntest {global_idx}/{len(test_questions)}: {test_case['question']}")

                try:
                    # obtient la réponse
                    result = rag_system.query(test_case["question"])

                    # évalue avec métriques basiques
                    result_data = await evaluate_response(evaluator, result, test_case)
                    batch_results.append(result_data)

                    # affiche les résultats
                    print(f"type de recherche: {result.get('search_type', 'semantic')}")
                    print(f"faithfulness: {result_data['faithfulness']:.3f}")
                    print(f"answer_relevancy: {result_data['answer_relevancy']:.3f}")
                    print(f"context_precision: {result_data['context_precision']:.3f}")
                    print(f"context_recall: {result_data['context_recall']:.3f}")

                except Exception as e:
                    print(f"erreur sur la question {global_idx}: {e}")
                    # ajoute un résultat vide en cas d'erreur
                    error_result = {
                        "question": test_case["question"],
                        "expected_type": test_case["type"],
                        "actual_type": "error",
                        "prediction": f"erreur: {str(e)}",
                        "reference": test_case["reference"],
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                    }
                    batch_results.append(error_result)

            # ajoute les résultats du lot
            all_results.extend(batch_results)

            # sauvegarde intermédiaire
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                batch_df.to_csv(output_dir / f"batch_{batch_idx}_results.csv", index=False)
                print(f"\nlot {batch_idx} sauvegardé: {len(batch_results)} résultats")

            # délai entre les lots (sauf le dernier)
            if batch_idx < len(batches):
                delay = 60  # 60 secondes entre les lots
                print(f"\nattente de {delay} secondes avant le prochain lot...")
                await asyncio.sleep(delay)

        # crée le dataframe final
        results_df = pd.DataFrame(all_results)

        # sauvegarde les résultats finaux
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        # génère les graphiques
        await evaluator.plot_results(results_df, output_dir, engaged_mode)

        # sauvegarde dans le dossier final
        save_results(results_df, output_dir, engaged_mode)

        # analyse des résultats
        print("\n" + "=" * 60)
        print("ANALYSE DÉTAILLÉE DES RÉSULTATS")
        print("=" * 60)

        # prépare le contenu pour le fichier texte
        report_content = []
        report_content.append("=" * 60)
        report_content.append("RAPPORT D'ÉVALUATION RAG POKÉMON")
        report_content.append("=" * 60)
        report_content.append(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"mode engagé: {'activé' if engaged_mode else 'désactivé'}")
        report_content.append(f"nombre total de questions: {len(results_df)}")
        report_content.append(f"nombre de lots: {len(batches)}")
        report_content.append("")

        # statistiques globales
        print("\nSTATISTIQUES GLOBALES:")
        print("-" * 40)
        report_content.append("STATISTIQUES GLOBALES:")
        report_content.append("-" * 40)

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        global_stats = (
            results_df[metrics]
            .agg(["mean", "std", "min", "max", "median"])
            .round(3)
        )
        print(global_stats)
        report_content.append(str(global_stats))
        report_content.append("")

        # analyse par type de question
        print("\nANALYSE PAR TYPE DE QUESTION:")
        print("-" * 40)
        report_content.append("ANALYSE PAR TYPE DE QUESTION:")
        report_content.append("-" * 40)

        type_stats = (
            results_df.groupby("expected_type")[metrics]
            .agg(["mean", "count"])
            .round(3)
        )
        print(type_stats)
        report_content.append(str(type_stats))
        report_content.append("")

        # analyse par type de recherche
        print("\nANALYSE PAR TYPE DE RECHERCHE:")
        print("-" * 40)
        report_content.append("ANALYSE PAR TYPE DE RECHERCHE:")
        report_content.append("-" * 40)

        search_stats = (
            results_df.groupby("actual_type")[metrics]
            .agg(["mean", "count"])
            .round(3)
        )
        print(search_stats)
        report_content.append(str(search_stats))
        report_content.append("")

        # distribution des scores
        print("\nDISTRIBUTION DES SCORES:")
        print("-" * 40)
        report_content.append("DISTRIBUTION DES SCORES:")
        report_content.append("-" * 40)

        for metric in metrics:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")

            excellent = len(results_df[results_df[metric] >= 0.9])
            good = len(
                results_df[(results_df[metric] >= 0.7) & (results_df[metric] < 0.9)]
            )
            medium = len(
                results_df[(results_df[metric] >= 0.5) & (results_df[metric] < 0.7)]
            )
            poor = len(results_df[results_df[metric] < 0.5])
            total = len(results_df)

            print(
                f"  excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)"
            )
            print(f"  bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)")
            print(f"  moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)")
            print(f"  faible (<0.5): {poor} questions ({poor/total*100:.1f}%)")

            report_content.append(
                f"  excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)"
            )
            report_content.append(
                f"  bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)"
            )
            report_content.append(
                f"  moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)"
            )
            report_content.append(
                f"  faible (<0.5): {poor} questions ({poor/total*100:.1f}%)"
            )

        # corrélations entre métriques
        print("\nCORRÉLATIONS ENTRE MÉTRIQUES:")
        print("-" * 40)
        report_content.append("\nCORRÉLATIONS ENTRE MÉTRIQUES:")
        report_content.append("-" * 40)

        correlation_matrix = results_df[metrics].corr().round(3)
        print(correlation_matrix)
        report_content.append(str(correlation_matrix))
        report_content.append("")

        # questions avec les meilleurs scores
        print("\nTOP 3 QUESTIONS PAR MÉTRIQUE:")
        print("-" * 40)
        report_content.append("TOP 3 QUESTIONS PAR MÉTRIQUE:")
        report_content.append("-" * 40)

        for metric in metrics:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")

            top_3 = results_df.nlargest(3, metric)[["question", metric]]
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                line = f"  {i}. {row['question'][:60]}... (score: {row[metric]:.3f})"
                print(line)
                report_content.append(line)

        # questions avec les plus mauvais scores
        print("\nQUESTIONS AVEC LES PLUS MAUVAIS SCORES:")
        print("-" * 40)
        report_content.append("\nQUESTIONS AVEC LES PLUS MAUVAIS SCORES:")
        report_content.append("-" * 40)

        for metric in metrics:
            print(f"\n{metric.upper()} (plus bas):")
            report_content.append(f"\n{metric.upper()} (plus bas):")

            worst_3 = results_df.nsmallest(3, metric)[["question", metric]]
            for i, (_, row) in enumerate(worst_3.iterrows(), 1):
                line = f"  {i}. {row['question'][:60]}... (score: {row[metric]:.3f})"
                print(line)
                report_content.append(line)

        # analyse des erreurs détaillée
        print("\nANALYSE DÉTAILLÉE DES ERREURS:")
        print("-" * 40)
        report_content.append("\nANALYSE DÉTAILLÉE DES ERREURS:")
        report_content.append("-" * 40)

        # questions avec faible faithfulness
        low_faithfulness = results_df[results_df["faithfulness"] < 0.7]
        if not low_faithfulness.empty:
            print(f"\nquestions avec faible faithfulness (<0.7): {len(low_faithfulness)}")
            print(
                f"moyenne faithfulness pour ces questions: {low_faithfulness['faithfulness'].mean():.3f}"
            )
            report_content.append(
                f"\nquestions avec faible faithfulness (<0.7): {len(low_faithfulness)}"
            )
            report_content.append(
                f"moyenne faithfulness pour ces questions: {low_faithfulness['faithfulness'].mean():.3f}"
            )

            for _, row in low_faithfulness.iterrows():
                print(f"\n  question: {row['question']}")
                print(f"  prédiction: {row['prediction'][:100]}...")
                print(f"  référence: {row['reference'][:100]}...")
                print(f"  score faithfulness: {row['faithfulness']:.3f}")

                report_content.append(f"\n  question: {row['question']}")
                report_content.append(f"  prédiction: {row['prediction'][:100]}...")
                report_content.append(f"  référence: {row['reference'][:100]}...")
                report_content.append(f"  score faithfulness: {row['faithfulness']:.3f}")
        else:
            print("toutes les questions ont une bonne faithfulness (≥0.7)")
            report_content.append("toutes les questions ont une bonne faithfulness (≥0.7)")

        # questions avec faible answer_relevancy
        low_relevancy = results_df[results_df["answer_relevancy"] < 0.5]
        if not low_relevancy.empty:
            print(f"\nquestions avec faible answer_relevancy (<0.5): {len(low_relevancy)}")
            print(f"moyenne answer_relevancy pour ces questions: {low_relevancy['answer_relevancy'].mean():.3f}")
            report_content.append(f"\nquestions avec faible answer_relevancy (<0.5): {len(low_relevancy)}")
            report_content.append(
                f"moyenne answer_relevancy pour ces questions: {low_relevancy['answer_relevancy'].mean():.3f}"
            )

        # résumé des performances
        print("\nRÉSUMÉ DES PERFORMANCES:")
        print("-" * 40)
        report_content.append("\nRÉSUMÉ DES PERFORMANCES:")
        report_content.append("-" * 40)

        summary_lines = [
            f"mode engagé: {'activé' if engaged_mode else 'désactivé'}",
            f"nombre total de questions: {len(results_df)}",
            f"faithfulness moyen: {results_df['faithfulness'].mean():.3f} ± {results_df['faithfulness'].std():.3f}",
            f"answer_relevancy moyen: {results_df['answer_relevancy'].mean():.3f} ± {results_df['answer_relevancy'].std():.3f}",
            f"context_precision moyen: {results_df['context_precision'].mean():.3f} ± {results_df['context_precision'].std():.3f}",
            f"context_recall moyen: {results_df['context_recall'].mean():.3f} ± {results_df['context_recall'].std():.3f}",
        ]

        for line in summary_lines:
            print(line)
            report_content.append(line)

        # sauvegarde le rapport dans un fichier texte
        mode_suffix = "_engaged" if engaged_mode else ""
        report_filename = f"evaluation_report{mode_suffix}.txt"
        report_path = Path("evaluation_results") / report_filename

        # crée le dossier s'il n'existe pas
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))

        print(f"\nrapport détaillé sauvegardé: {report_path}")
        report_content.append(f"\nrapport détaillé sauvegardé: {report_path}")

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
    """crée un fichier de questions d'exemple pour les tests (20 questions variées)."""
    import json

    sample_questions = [
        # Statistiques de base
        {
            "question": "donne-moi exactement les statistiques de base de pikachu (pv, attaque, défense, attaque spéciale, défense spéciale, vitesse).",
            "reference": "pikachu a 35 points de vie, 55 d'attaque, 40 de défense, 50 d'attaque spéciale, 50 de défense spéciale et 90 de vitesse.",
            "type": "statistics",
        },
        {
            "question": "donne-moi exactement les statistiques de base de bulbizarre (pv, attaque, défense, attaque spéciale, défense spéciale, vitesse).",
            "reference": "bulbizarre a 45 points de vie, 49 d'attaque, 49 de défense, 65 d'attaque spéciale, 65 de défense spéciale et 45 de vitesse.",
            "type": "statistics",
        },
        {
            "question": "donne-moi exactement les statistiques de base de salamèche (pv, attaque, défense, attaque spéciale, défense spéciale, vitesse).",
            "reference": "salamèche a 39 points de vie, 52 d'attaque, 43 de défense, 60 d'attaque spéciale, 50 de défense spéciale et 65 de vitesse.",
            "type": "statistics",
        },
        {
            "question": "donne-moi exactement les statistiques de base de tortank (pv, attaque, défense, attaque spéciale, défense spéciale, vitesse).",
            "reference": "tortank a 79 points de vie, 83 d'attaque, 100 de défense, 85 d'attaque spéciale, 105 de défense spéciale et 78 de vitesse.",
            "type": "statistics",
        },
        # Types et capacités
        {
            "question": "quels sont les deux types de dracaufeu et ses capacités spéciales selon le pokédex ?",
            "reference": "dracaufeu est de type feu et vol. ses capacités spéciales sont blaze et solar power.",
            "type": "statistics",
        },
        {
            "question": "quels sont les deux types de florizarre et ses capacités spéciales selon le pokédex ?",
            "reference": "florizarre est de type plante et poison. ses capacités spéciales sont overgrow et chlorophyll.",
            "type": "statistics",
        },
        # Listes par type
        {
            "question": "donne la liste complète des pokémon de type feu de la première génération.",
            "reference": "les pokémon de type feu de la première génération sont salamèche, reptincel, dracaufeu, caninos, arcanin, ponyta, galopa, goupix, feunard, magmar et pyroli.",
            "type": "categorization",
        },
        {
            "question": "donne la liste complète des pokémon de type eau de la première génération.",
            "reference": "les pokémon de type eau de la première génération incluent carapuce, carabaffe, tortank, magicarpe, léviator, ptitard, têtarte, tartard, psykokwak, akwakwak, poissirène, poissoroy, hypotrempe, hypocéan, lamantine, otaria, lokhlass, stari, staross, tentacool, tentacruel, krabboss, krabby, kokiyas, et d'autres.",
            "type": "categorization",
        },
        # Statut légendaire/mythique
        {
            "question": "cites trois pokémon légendaires de la première génération.",
            "reference": "trois pokémon légendaires de la première génération sont artikodin, électhor et sulfura.",
            "type": "categorization",
        },
        {
            "question": "donne un exemple de pokémon mythique de la première génération.",
            "reference": "mew est un pokémon mythique de la première génération.",
            "type": "categorization",
        },
        # Descriptions Poképédia
        {
            "question": "décris le comportement de ronflex selon poképédia.",
            "reference": "ronflex passe la majeure partie de son temps à dormir et à manger. il est réputé pour sa grande paresse et son appétit insatiable.",
            "type": "description",
        },
        {
            "question": "dans quel habitat naturel vit principalement bulbizarre selon poképédia ?",
            "reference": "bulbizarre vit principalement dans les forêts et les prairies.",
            "type": "description",
        },
        {
            "question": "quelle est la couleur de pikachu selon le pokédex ?",
            "reference": "pikachu est de couleur jaune.",
            "type": "description",
        },
        # Évolution
        {
            "question": "en quoi évolue salamèche ?",
            "reference": "salamèche évolue en reptincel.",
            "type": "evolution",
        },
        {
            "question": "en quoi évolue carapuce ?",
            "reference": "carapuce évolue en carabaffe.",
            "type": "evolution",
        },
        # Comparaisons
        {
            "question": "lequel a le plus de points de vie, ronflex ou pikachu ?",
            "reference": "ronflex a plus de points de vie que pikachu.",
            "type": "comparison",
        },
        {
            "question": "lequel est plus rapide, pikachu ou tortank ?",
            "reference": "pikachu est plus rapide que tortank.",
            "type": "comparison",
        },
        # Listes par habitat/couleur
        {
            "question": "donne la liste des pokémon de couleur jaune de la première génération.",
            "reference": "les pokémon de couleur jaune de la première génération incluent pikachu, raichu, léviator chromatique, etc.",
            "type": "categorization",
        },
        {
            "question": "donne la liste des pokémon vivant dans les forêts selon le pokédex.",
            "reference": "les pokémon vivant dans les forêts incluent bulbizarre, chenipan, aspicot, pikachu, etc.",
            "type": "categorization",
        },
        # Capacités spéciales
        {
            "question": "quelles sont les capacités spéciales de salamèche ?",
            "reference": "salamèche a la capacité blaze qui augmente la puissance des attaques feu.",
            "type": "statistics",
        },
        {
            "question": "quelles sont les capacités spéciales de ronflex ?",
            "reference": "ronflex a les capacités immunité et épais gras.",
            "type": "statistics",
        },
        # Liste par type normal
        {
            "question": "quels sont les pokémon de type normal de la première génération ?",
            "reference": "les pokémon de type normal de la première génération incluent rattata, rattatac, ronflex, miaouss, persian, piafabec, rapasdepic, noeunoeuf, noadkoko, kangourex, tauros, ditto, évoli, lippoutou, leveinard, et d'autres.",
            "type": "categorization",
        },
    ]

    # crée le dossier data s'il n'existe pas
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # sauvegarde le fichier
    with open(data_dir / "test_questions.json", "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, ensure_ascii=False, indent=2)

    print(f"fichier de questions d'exemple créé: {data_dir / 'test_questions.json'}")


async def run_evaluation(dataset_path: Path | None = None, engaged_mode: bool = False) -> None:
    """lance l'évaluation rag complète (alias pour la compatibilité)."""
    await run_evaluation_in_batches(dataset_path, batch_size=10, engaged_mode=engaged_mode)


async def resume_evaluation(dataset_path: Path | None = None, start_from: int = 0, batch_size: int = 10, engaged_mode: bool = False) -> None:
    """reprend l'évaluation à partir d'un certain point."""
    print("initialisation...")
    print(f"mode engagé: {'activé' if engaged_mode else 'désactivé'}")
    
    rag_system = RAGSystem(engaged_mode=engaged_mode)
    evaluator = RAGEvaluator()

    # charge les documents
    print("chargement des documents...")
    documents = create_pokemon_documents()
    rag_system.embed_documents(documents)

    # charge les questions
    if dataset_path is None:
        raise ValueError("chemin du jeu de questions non fourni")

    test_questions = load_questions(dataset_path)
    print(f"total questions à évaluer: {len(test_questions)}")
    print(f"reprise à partir de la question {start_from + 1}")

    # filtre les questions à partir du point de reprise
    remaining_questions = test_questions[start_from:]
    
    # découpe en lots
    batches = [remaining_questions[i:i + batch_size] for i in range(0, len(remaining_questions), batch_size)]
    print(f"découpage en {len(batches)} lots de {batch_size} questions")

    # prépare les résultats
    all_results = []
    output_dir = "evaluation_results"

    try:
        # crée le dossier temporaire
        output_dir = Path(tempfile.mkdtemp(prefix="eval_results_"))

        # traite chaque lot
        for batch_idx, batch in enumerate(batches, 1):
            print(f"\n{'='*60}")
            print(f"LOT {batch_idx}/{len(batches)} ({len(batch)} questions)")
            print(f"{'='*60}")

            batch_results = []

            # évalue chaque question du lot
            for i, test_case in enumerate(batch, 1):
                global_idx = start_from + (batch_idx - 1) * batch_size + i
                print(f"\ntest {global_idx}/{len(test_questions)}: {test_case['question']}")

                try:
                    # obtient la réponse
                    result = rag_system.query(test_case["question"])

                    # évalue avec métriques basiques
                    result_data = await evaluate_response(evaluator, result, test_case)
                    batch_results.append(result_data)

                    # affiche les résultats
                    print(f"type de recherche: {result.get('search_type', 'semantic')}")
                    print(f"faithfulness: {result_data['faithfulness']:.3f}")
                    print(f"answer_relevancy: {result_data['answer_relevancy']:.3f}")
                    print(f"context_precision: {result_data['context_precision']:.3f}")
                    print(f"context_recall: {result_data['context_recall']:.3f}")

                except Exception as e:
                    print(f"erreur sur la question {global_idx}: {e}")
                    # ajoute un résultat vide en cas d'erreur
                    error_result = {
                        "question": test_case["question"],
                        "expected_type": test_case["type"],
                        "actual_type": "error",
                        "prediction": f"erreur: {str(e)}",
                        "reference": test_case["reference"],
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                    }
                    batch_results.append(error_result)

            # ajoute les résultats du lot
            all_results.extend(batch_results)

            # sauvegarde intermédiaire
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                batch_df.to_csv(output_dir / f"batch_{batch_idx}_results.csv", index=False)
                print(f"\nlot {batch_idx} sauvegardé: {len(batch_results)} résultats")

            # délai entre les lots (sauf le dernier)
            if batch_idx < len(batches):
                delay = 60  # 60 secondes entre les lots
                print(f"\nattente de {delay} secondes avant le prochain lot...")
                await asyncio.sleep(delay)

        # crée le dataframe final
        results_df = pd.DataFrame(all_results)

        # sauvegarde les résultats finaux
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        # génère les graphiques
        await evaluator.plot_results(results_df, output_dir, engaged_mode)

        # sauvegarde dans le dossier final
        save_results(results_df, output_dir, engaged_mode)

        print(f"\névaluation terminée: {len(results_df)} questions traitées")

    finally:
        # nettoie le dossier temporaire
        if output_dir and output_dir.exists():
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as e:
                print(f"erreur de nettoyage : {e}")


if __name__ == "__main__":
    # enregistre la fonction de nettoyage
    atexit.register(cleanup)

    # affiche l'aide si demandé
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
ÉVALUATION RAG POKÉMON
======================

Usage:
  python src/evaluate_rag.py [fichier_questions] [question_debut] [options]

Arguments:
  fichier_questions    Chemin vers le fichier JSON contenant les questions de test
                       (défaut: data/test_questions.json)
  question_debut       Numéro de la question à partir de laquelle reprendre l'évaluation
                       (optionnel, pour reprendre une évaluation interrompue)

Options:
  --engaged            Active le mode engagé pour des réponses plus détaillées
  --help, -h           Affiche cette aide

Exemples:
  python src/evaluate_rag.py                           # Évaluation normale avec fichier par défaut
  python src/evaluate_rag.py data/test_questions.json  # Évaluation avec fichier spécifique
  python src/evaluate_rag.py --engaged                 # Évaluation en mode engagé
  python src/evaluate_rag.py data/test_questions.json 5 --engaged  # Reprendre à la question 5 en mode engagé

Le mode engagé utilise des prompts plus détaillés et récupère plus de contexte pour des réponses plus complètes.
""")
        sys.exit(0)

    # parse les arguments
    engaged_mode = "--engaged" in sys.argv
    if engaged_mode:
        sys.argv.remove("--engaged")  # retire l'argument pour ne pas interférer avec les autres

    # chemin du jeu de questions
    if len(sys.argv) > 1:
        dataset = Path(sys.argv[1])
    else:
        # utilise un fichier par défaut s'il existe
        default_dataset = Path("data/test_questions.json")
        if default_dataset.exists():
            dataset = default_dataset
            print(f"utilisation du fichier de questions par défaut: {dataset}")
        else:
            print(
                "aucun fichier de questions fourni et aucun fichier par défaut trouvé."
            )
            print("création d'un fichier de questions d'exemple...")
            create_sample_questions()
            dataset = default_dataset

    # vérifie s'il y a un argument pour reprendre l'évaluation
    start_from = 0
    if len(sys.argv) > 2:
        try:
            start_from = int(sys.argv[2])
            print(f"reprise de l'évaluation à partir de la question {start_from + 1}")
            asyncio.run(resume_evaluation(dataset, start_from, engaged_mode=engaged_mode))
        except ValueError:
            print("argument de reprise invalide, lancement de l'évaluation complète")
            asyncio.run(run_evaluation_in_batches(dataset, engaged_mode=engaged_mode))
    else:
        # lance l'évaluation complète
        asyncio.run(run_evaluation_in_batches(dataset, engaged_mode=engaged_mode))
