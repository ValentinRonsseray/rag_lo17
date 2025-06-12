"""
Evaluation module for the RAG system.
"""

import json
from typing import List, Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
import asyncio
import os
from dotenv import load_dotenv

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator
from langchain_google_genai import ChatGoogleGenerativeAI

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

        # Initialize Gemini LLM for RAGAS metrics
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.0
        )

        # Initialize RAG metrics with LLM
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        self.context_entities_recall = ContextEntityRecall(llm=self.llm)
        self.noise_sensitivity = NoiseSensitivity(llm=self.llm)
        self.response_relevancy = ResponseRelevancy(llm=self.llm)
        self.faithfulness_metric = Faithfulness(llm=self.llm)

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

        # Build a SingleTurnSample for ragas metrics
        sample = SingleTurnSample(
            prompt="",
            response=prediction,
            reference=reference,
            context=context,
        )

        try:
            # Compute RAG metrics via single-turn scoring
            cp_score = await self.context_precision.single_turn_ascore(sample)
            cr_score = await self.context_recall.single_turn_ascore(sample)
            cer_score = await self.context_entities_recall.single_turn_ascore(sample)
            ns_score = await self.noise_sensitivity.single_turn_ascore(sample)
            rr_score = await self.response_relevancy.single_turn_ascore(sample)
            faith_score = await self.faithfulness_metric.single_turn_ascore(sample)
        except Exception as e:
            print(f"Erreur lors du calcul des métriques RAGAS: {e}")
            # Utiliser des valeurs par défaut en cas d'erreur
            cp_score = 0.0
            cr_score = 0.0
            cer_score = 0.0
            ns_score = 0.0
            rr_score = 0.0
            faith_score = 0.0

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
