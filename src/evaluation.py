"""
Evaluation module for the RAG system.
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

# Load environment variables
load_dotenv()

def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def exact_match_score(prediction: str, reference: str) -> float:
    """Calculate exact match score (0 or 1)."""
    return float(normalize_text(prediction) == normalize_text(reference))

def f1_score_text(prediction: str, reference: str) -> float:
    """Calculate F1 score for text."""
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
    """Calculate the proportion of words in the prediction that appear in the context."""
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
        """Initialize the RAG evaluator with lightweight metrics."""

        # Default string-distance evaluator if none provided
        self.llm_evaluator = llm_evaluator or load_evaluator("string_distance")

    async def evaluate_response(
        self,
        prediction: str,
        reference: str,
        context: List[str]
    ) -> Dict[str, float]:
        """Evaluate a single response.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            context: Retrieved context (list of strings)
        
        Returns:
            Dictionary of scores
        """
        # Built-in string metrics
        em_score = exact_match_score(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        # Simple overlap metric between the answer and the retrieved context
        overlap = context_overlap_score(prediction, context)
        faith_score = overlap

        return {
            "exact_match": em_score,
            "f1_score": f1,
            "context_overlap": overlap,
            "faithfulness": faith_score,
        }

    async def evaluate_dataset(
        self,
        predictions: List[str],
        references: List[str],
        contexts: List[List[str]]
    ) -> pd.DataFrame:
        """Evaluate a dataset of responses."""
        results = []
        for pred, ref, ctx in zip(predictions, references, contexts):
            scores = await self.evaluate_response(pred, ref, ctx)
            results.append(scores)
        return pd.DataFrame(results)

    async def plot_results(
        self,
        results_df: pd.DataFrame,
        output_dir: Path
    ):
        """Plot evaluation results for all metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sélectionner uniquement les colonnes numériques pour les graphiques
        numeric_columns = results_df.select_dtypes(include=['float64', 'int64']).columns
        num = len(numeric_columns)
        cols = min(4, num)
        rows = (num + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes_list = axes.flatten()

        for idx, metric in enumerate(numeric_columns):
            ax = axes_list[idx]
            ax.hist(results_df[metric], bins=10)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")

        # Hide unused subplots
        for idx in range(num, len(axes_list)):
            axes_list[idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png")
        plt.close(fig)

        # Save results to CSV
        results_df.to_csv(output_dir / "eval_metrics.csv", index=False)

        # Print summary
        print("\nEvaluation Summary:")
        print("\nMétriques numériques:")
        print(results_df[numeric_columns].mean())
        
        # Afficher les résultats non numériques
        non_numeric_columns = results_df.select_dtypes(exclude=['float64', 'int64']).columns
        if len(non_numeric_columns) > 0:
            print("\nRésultats non numériques:")
            for col in non_numeric_columns:
                print(f"\n{col}:")
                print(results_df[col].value_counts())
