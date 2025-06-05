"""
Evaluation module for the RAG system.
"""

import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import re

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator

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
    """Calculate exact match score.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        Exact match score (0 or 1)
    """
    return float(normalize_text(prediction) == normalize_text(reference))

def f1_score_text(prediction: str, reference: str) -> float:
    """Calculate F1 score for text.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        F1 score
    """
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
    def __init__(self, llm_evaluator: StringEvaluator = None):
        """Initialize the RAG evaluator.
        
        Args:
            llm_evaluator: LangChain evaluator for LLM-based scoring
        """
        self.llm_evaluator = llm_evaluator or load_evaluator("string_distance")
        
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
            context: Retrieved context
            
        Returns:
            Dictionary of scores
        """
        # Calculate string-based metrics
        em_score = exact_match_score(prediction, reference)
        f1 = f1_score_text(prediction, reference)
        
        # Calculate LLM-based metrics
        faithfulness = self.llm_evaluator.evaluate_strings(
            prediction=prediction,
            reference=reference,
            input="\n".join(context)
        )
        
        return {
            "exact_match": em_score,
            "f1_score": f1,
            "faithfulness": faithfulness["score"]
        }
        
    def evaluate_dataset(
        self,
        predictions: List[str],
        references: List[str],
        contexts: List[List[str]]
    ) -> pd.DataFrame:
        """Evaluate a dataset of responses.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            contexts: List of retrieved contexts
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        for pred, ref, ctx in zip(predictions, references, contexts):
            scores = self.evaluate_response(pred, ref, ctx)
            results.append(scores)
            
        return pd.DataFrame(results)
    
    def plot_results(self, results_df: pd.DataFrame, output_dir: Path):
        """Plot evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(["exact_match", "f1_score", "faithfulness"]):
            axes[i].hist(results_df[metric], bins=10)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_xlabel("Score")
            axes[i].set_ylabel("Count")
            
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png")
        plt.close()
        
        # Save results
        results_df.to_csv(output_dir / "eval_metrics.csv", index=False)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(results_df.mean()) 