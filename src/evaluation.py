from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from rapidfuzz import fuzz


def _normalize(text: str) -> str:
    """nettoie le texte pour les comparaisons."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    """découpe le texte en tokens simples."""
    return _normalize(text).split()


def f1_score(prediction: str, reference: str) -> float:
    """calcule le score f1 entre la prédiction et la référence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens).intersection(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def similarity(prediction: str, reference: str) -> float:
    """mesure de similarité rapide entre deux textes."""
    return fuzz.token_set_ratio(prediction, reference) / 100.0


def _context_tokens(context: List[str]) -> List[str]:
    tokens: List[str] = []
    for ctx in context:
        tokens.extend(_tokenize(ctx))
    return tokens


def context_precision(context: List[str], reference: str) -> float:
    """précision du contexte par rapport à la référence."""
    ctx_tokens = _context_tokens(context)
    ref_tokens = _tokenize(reference)
    if not ctx_tokens:
        return 0.0
    common = set(ctx_tokens).intersection(ref_tokens)
    return len(common) / len(ctx_tokens)


def context_recall(context: List[str], reference: str) -> float:
    """rappel du contexte par rapport à la référence."""
    ctx_tokens = _context_tokens(context)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    common = set(ref_tokens).intersection(ctx_tokens)
    return len(common) / len(ref_tokens)


def faithfulness(prediction: str, context: List[str]) -> float:
    """mesure la proportion de la réponse présente dans le contexte."""
    pred_tokens = _tokenize(prediction)
    ctx_tokens = _context_tokens(context)
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens).intersection(ctx_tokens)
    return len(common) / len(pred_tokens)


class RAGEvaluator:
    """évaluateur utilisant des métriques inspirées de ragas."""

    def __init__(self) -> None:
        pass

    async def evaluate_response(
        self, prediction: str, reference: str, context: List[str]
    ) -> Dict[str, float]:
        """évalue une paire prédiction/référence avec son contexte."""
        return {
            "answer_f1": f1_score(prediction, reference),
            "answer_similarity": similarity(prediction, reference),
            "context_precision": context_precision(context, reference),
            "context_recall": context_recall(context, reference),
            "faithfulness": faithfulness(prediction, context),
        }

    async def evaluate_dataset(
        self, predictions: List[str], references: List[str], contexts: List[List[str]]
    ) -> pd.DataFrame:
        """évalue un ensemble de prédictions."""
        rows = []
        for pred, ref, ctx in zip(predictions, references, contexts):
            rows.append(await self.evaluate_response(pred, ref, ctx))
        return pd.DataFrame(rows)

    async def plot_results(self, results_df: pd.DataFrame, output_dir: Path) -> None:
        """crée un histogramme pour chaque métrique."""
        output_dir.mkdir(parents=True, exist_ok=True)
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

        for idx in range(num, len(axes_list)):
            axes_list[idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png")
        plt.close(fig)
        results_df.to_csv(output_dir / "eval_metrics.csv", index=False)
        print("\nrésumé de l'évaluation :")
        print(results_df[numeric_columns].mean())
