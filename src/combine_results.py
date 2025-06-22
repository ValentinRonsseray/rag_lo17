"""script pour combiner les résultats d'évaluation et générer un rapport final."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# ajout du répertoire racine au path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.evaluation import RAGEvaluator


def combine_evaluation_results() -> None:
    """combine tous les résultats d'évaluation et génère un rapport final."""
    print("combinaison des résultats d'évaluation...")
    
    # charge les questions originales pour avoir le contexte complet
    questions_file = Path("data/test_questions.json")
    if not questions_file.exists():
        print("fichier de questions non trouvé")
        return
    
    with open(questions_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    
    print(f"total questions dans le jeu de test: {len(all_questions)}")
    
    # cherche tous les fichiers de résultats
    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        print("dossier de résultats non trouvé")
        return
    
    # combine tous les fichiers csv
    all_results = []
    csv_files = list(results_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        if csv_file.name != "eval_metrics.csv":  # exclut le fichier de métriques
            try:
                df = pd.read_csv(csv_file)
                all_results.append(df)
                print(f"fichier chargé: {csv_file.name} ({len(df)} résultats)")
            except Exception as e:
                print(f"erreur lors du chargement de {csv_file}: {e}")
    
    if not all_results:
        print("aucun résultat trouvé")
        return
    
    # combine tous les résultats
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"total résultats combinés: {len(combined_df)}")
    
    # supprime les doublons basés sur la question
    combined_df = combined_df.drop_duplicates(subset=['question'], keep='first')
    print(f"après suppression des doublons: {len(combined_df)} résultats")
    
    # vérifie la couverture
    covered_questions = set(combined_df['question'].tolist())
    all_question_texts = set(q['question'] for q in all_questions)
    
    missing_questions = all_question_texts - covered_questions
    if missing_questions:
        print(f"\nquestions manquantes ({len(missing_questions)}):")
        for q in missing_questions:
            print(f"  - {q[:60]}...")
    
    # sauvegarde le résultat combiné
    combined_file = results_dir / "combined_evaluation_results.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\nrésultats combinés sauvegardés: {combined_file}")
    
    # génère le rapport final
    generate_final_report(combined_df, len(all_questions), len(missing_questions))


def generate_final_report(results_df: pd.DataFrame, total_questions: int, missing_questions: int) -> None:
    """génère un rapport final détaillé."""
    print("\n" + "=" * 60)
    print("RAPPORT FINAL D'ÉVALUATION RAG POKÉMON")
    print("=" * 60)
    
    # prépare le contenu du rapport
    report_content = []
    report_content.append("=" * 60)
    report_content.append("RAPPORT FINAL D'ÉVALUATION RAG POKÉMON")
    report_content.append("=" * 60)
    report_content.append(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"nombre total de questions dans le jeu de test: {total_questions}")
    report_content.append(f"nombre de questions évaluées: {len(results_df)}")
    report_content.append(f"nombre de questions manquantes: {missing_questions}")
    report_content.append(f"taux de couverture: {len(results_df)/total_questions*100:.1f}%")
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
    
    # filtre les erreurs
    valid_results = results_df[results_df['actual_type'] != 'error']
    error_count = len(results_df) - len(valid_results)
    
    if len(valid_results) > 0:
        global_stats = (
            valid_results[metrics]
            .agg(["mean", "std", "min", "max", "median"])
            .round(3)
        )
        print(global_stats)
        report_content.append(str(global_stats))
        report_content.append("")
        
        print(f"erreurs d'api: {error_count}")
        report_content.append(f"erreurs d'api: {error_count}")
        report_content.append("")
    else:
        print("aucun résultat valide trouvé")
        report_content.append("aucun résultat valide trouvé")
        report_content.append("")
    
    # analyse par type de question
    if len(valid_results) > 0:
        print("\nANALYSE PAR TYPE DE QUESTION:")
        print("-" * 40)
        report_content.append("ANALYSE PAR TYPE DE QUESTION:")
        report_content.append("-" * 40)
        
        type_stats = (
            valid_results.groupby("expected_type")[metrics]
            .agg(["mean", "count"])
            .round(3)
        )
        print(type_stats)
        report_content.append(str(type_stats))
        report_content.append("")
        
        # distribution des scores
        print("\nDISTRIBUTION DES SCORES:")
        print("-" * 40)
        report_content.append("DISTRIBUTION DES SCORES:")
        report_content.append("-" * 40)
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")
            
            excellent = len(valid_results[valid_results[metric] >= 0.9])
            good = len(valid_results[(valid_results[metric] >= 0.7) & (valid_results[metric] < 0.9)])
            medium = len(valid_results[(valid_results[metric] >= 0.5) & (valid_results[metric] < 0.7)])
            poor = len(valid_results[valid_results[metric] < 0.5])
            total = len(valid_results)
            
            if total > 0:
                print(f"  excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)")
                print(f"  bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)")
                print(f"  moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)")
                print(f"  faible (<0.5): {poor} questions ({poor/total*100:.1f}%)")
                
                report_content.append(f"  excellent (≥0.9): {excellent} questions ({excellent/total*100:.1f}%)")
                report_content.append(f"  bon (0.7-0.9): {good} questions ({good/total*100:.1f}%)")
                report_content.append(f"  moyen (0.5-0.7): {medium} questions ({medium/total*100:.1f}%)")
                report_content.append(f"  faible (<0.5): {poor} questions ({poor/total*100:.1f}%)")
        
        # top 5 questions par métrique
        print("\nTOP 5 QUESTIONS PAR MÉTRIQUE:")
        print("-" * 40)
        report_content.append("\nTOP 5 QUESTIONS PAR MÉTRIQUE:")
        report_content.append("-" * 40)
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            report_content.append(f"\n{metric.upper()}:")
            
            top_5 = valid_results.nlargest(5, metric)[["question", metric]]
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                line = f"  {i}. {row['question'][:60]}... (score: {row[metric]:.3f})"
                print(line)
                report_content.append(line)
    
    # résumé final
    print("\nRÉSUMÉ FINAL:")
    print("-" * 40)
    report_content.append("\nRÉSUMÉ FINAL:")
    report_content.append("-" * 40)
    
    summary_lines = [
        f"jeu de test: {total_questions} questions",
        f"évaluées avec succès: {len(results_df)} questions",
        f"erreurs d'api: {error_count} questions",
        f"taux de couverture: {len(results_df)/total_questions*100:.1f}%",
    ]
    
    if len(valid_results) > 0:
        summary_lines.extend([
            f"faithfulness moyen: {valid_results['faithfulness'].mean():.3f} ± {valid_results['faithfulness'].std():.3f}",
            f"answer_relevancy moyen: {valid_results['answer_relevancy'].mean():.3f} ± {valid_results['answer_relevancy'].std():.3f}",
            f"context_precision moyen: {valid_results['context_precision'].mean():.3f} ± {valid_results['context_precision'].std():.3f}",
            f"context_recall moyen: {valid_results['context_recall'].mean():.3f} ± {valid_results['context_recall'].std():.3f}",
        ])
    
    for line in summary_lines:
        print(line)
        report_content.append(line)
    
    # sauvegarde le rapport final
    report_file = Path("evaluation_results") / "final_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    
    print(f"\nrapport final sauvegardé: {report_file}")


if __name__ == "__main__":
    combine_evaluation_results() 