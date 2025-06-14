"""
module d'évaluation du système rag
"""

import json
from typing import List, Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator

# charge les variables d'environnement
load_dotenv()


def normalize_text(text: str) -> str:
    """normalise le texte pour comparaison"""
    # minuscules
    text = text.lower()
    # supprime la ponctuation
    text = re.sub(r"[^\w\s]", "", text)
    # supprime les espaces en trop
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, reference: str) -> float:
    """calcule le score de correspondance exacte"""
    return float(normalize_text(prediction) == normalize_text(reference))


def f1_score_text(prediction: str, reference: str) -> float:
    """calcule le score f1 pour le texte"""
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens.intersection(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def context_overlap_score(prediction: str, context: List[str]) -> float:
    """calcule le recouvrement avec le contexte"""
    pred_tokens = set(normalize_text(prediction).split())
    context_tokens: set[str] = set()
    for ctx in context:
        context_tokens.update(normalize_text(ctx).split())
    if not pred_tokens:
        return 0.0
    overlap = pred_tokens.intersection(context_tokens)
    return len(overlap) / len(pred_tokens)


class RAGEvaluator:
    def __init__(self, llm_evaluator: StringEvaluator = None):
        """init de l'évaluateur rag"""

        # évaluateur par défaut
        self.llm_evaluator = llm_evaluator or load_evaluator("string_distance")

    async def evaluate_response(
        self, prediction: str, reference: str, context: List[str]
    ) -> Dict[str, float]:
        """évalue une réponse"""
        # métriques de base
        em_score = exact_match_score(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        # métrique de recouvrement
        overlap = context_overlap_score(prediction, context)
        faith_score = overlap

        return {
            "exact_match": em_score,
            "f1_score": f1,
            "context_overlap": overlap,
            "faithfulness": faith_score,
        }

    async def evaluate_dataset(
        self, predictions: List[str], references: List[str], contexts: List[List[str]]
    ) -> pd.DataFrame:
        """évalue un jeu de données"""
        results = []
        for pred, ref, ctx in zip(predictions, references, contexts):
            scores = await self.evaluate_response(pred, ref, ctx)
            results.append(scores)
        return pd.DataFrame(results)

    async def plot_results(self, results_df: pd.DataFrame, output_dir: Path):
        """trace les résultats d'évaluation"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # colonnes numériques pour les graphiques
        numeric_columns = results_df.select_dtypes(include=["float64", "int64"]).columns
        num = len(numeric_columns)
        cols = min(4, num)
        rows = (num + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes_list = axes.flatten()

        for idx, metric in enumerate(numeric_columns):
            ax = axes_list[idx]
            ax.hist(results_df[metric], bins=10)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("score")
            ax.set_ylabel("compte")

        # cache les sous-graphiques inutilisés
        for idx in range(num, len(axes_list)):
            axes_list[idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png")
        plt.close(fig)

        # sauvegarde les résultats
        results_df.to_csv(output_dir / "eval_metrics.csv", index=False)

        # affiche le résumé
        print("\nrésumé de l'évaluation:")
        print("\nmétriques numériques:")
        print(results_df[numeric_columns].mean())

        # affiche les résultats non numériques
        non_numeric_columns = results_df.select_dtypes(
            exclude=["float64", "int64"]
        ).columns
        if len(non_numeric_columns) > 0:
            print("\nrésultats non numériques:")
            for col in non_numeric_columns:
                print(f"\n{col}:")
                print(results_df[col].value_counts())
