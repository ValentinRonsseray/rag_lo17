"""
Evaluation module for the RAG system.
"""

import json
from typing import List, Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator

# Import new RAG metrics
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    ResponseRelevancy,
    Faithfulness,
)
from ragas import SingleTurnSample

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

class RAGEvaluator:
    def __init__(
        self,
        llm_evaluator: StringEvaluator = None
    ):
        """Initialize the RAG evaluator.
        
        Args:
            llm_evaluator: LangChain evaluator for LLM-based scoring
        """
        # Default string-distance evaluator if none provided
        self.llm_evaluator = llm_evaluator or load_evaluator("string_distance")

        # Initialize RAG metrics
        self.context_precision = ContextPrecision()
        self.context_recall = ContextRecall()
        self.context_entities_recall = ContextEntityRecall()
        self.noise_sensitivity = NoiseSensitivity()
        self.response_relevancy = ResponseRelevancy(llm=self.llm_evaluator)
        self.faithfulness_metric = Faithfulness(llm=self.llm_evaluator)

    def evaluate_response(
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

        # Build a SingleTurnSample for ragas metrics
        sample = SingleTurnSample(
            prompt="",
            response=prediction,
            reference=reference,
            context=context,
        )

        # Compute RAG metrics via single-turn scoring
        cp_score  = self.context_precision.single_turn_ascore(sample)
        cr_score  = self.context_recall.single_turn_ascore(sample)
        cer_score = self.context_entities_recall.single_turn_ascore(sample)
        ns_score  = self.noise_sensitivity.single_turn_ascore(sample)
        rr_score  = self.response_relevancy.single_turn_ascore(sample)
        faith_score = self.faithfulness_metric.single_turn_ascore(sample)

        return {
            "exact_match": em_score,
            "f1_score": f1,
            "context_precision": cp_score,
            "context_recall": cr_score,
            "context_entities_recall": cer_score,
            "noise_sensitivity": ns_score,
            "response_relevancy": rr_score,
            "faithfulness": faith_score,
        }

    def evaluate_dataset(
        self,
        predictions: List[str],
        references: List[str],
        contexts: List[List[str]]
    ) -> pd.DataFrame:
        """Evaluate a dataset of responses."""
        results = []
        for pred, ref, ctx in zip(predictions, references, contexts):
            scores = self.evaluate_response(pred, ref, ctx)
            results.append(scores)
        return pd.DataFrame(results)

    def plot_results(
        self,
        results_df: pd.DataFrame,
        output_dir: Path
    ):
        """Plot evaluation results for all metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dynamically plot each metric
        metrics = results_df.columns.tolist()
        num = len(metrics)
        cols = min(4, num)
        rows = (num + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes_list = axes.flatten()

        for idx, metric in enumerate(metrics):
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
        print(results_df.mean())
