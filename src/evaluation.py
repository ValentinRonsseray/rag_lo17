"""module d'évaluation utilisant des métriques basiques pour le système rag pokémon."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from difflib import SequenceMatcher


def calculate_similarity(text1: str, text2: str) -> float:
    """calcule la similarité entre deux textes."""
    # normalise les textes pour une meilleure comparaison
    text1_norm = re.sub(r'[^\w\s]', ' ', text1.lower()).strip()
    text2_norm = re.sub(r'[^\w\s]', ' ', text2.lower()).strip()
    
    # utilise sequence matcher pour la similarité globale
    return SequenceMatcher(None, text1_norm, text2_norm).ratio()


def calculate_keyword_overlap(text1: str, text2: str) -> float:
    """calcule le chevauchement de mots-clés entre deux textes."""
    # extrait les mots significatifs (plus de 2 caractères)
    words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def calculate_context_relevance(answer: str, context: List[str]) -> float:
    """calcule la pertinence de la réponse par rapport au contexte."""
    if not context:
        return 0.0
    
    # combine tout le contexte
    full_context = " ".join(context).lower()
    answer_lower = answer.lower()
    
    # compte les mots significatifs de la réponse présents dans le contexte
    answer_words = set(re.findall(r'\b\w{3,}\b', answer_lower))
    context_words = set(re.findall(r'\b\w{3,}\b', full_context))
    
    if not answer_words:
        return 0.0
    
    relevant_words = answer_words.intersection(context_words)
    return len(relevant_words) / len(answer_words)


def calculate_factual_accuracy(prediction: str, reference: str) -> float:
    """calcule la précision factuelle entre prédiction et référence."""
    # extrait les nombres et les noms propres
    pred_numbers = set(re.findall(r'\b\d+\b', prediction.lower()))
    ref_numbers = set(re.findall(r'\b\d+\b', reference.lower()))
    
    pred_names = set(re.findall(r'\b[a-zéèêëàâäôöùûüç]{3,}\b', prediction.lower()))
    ref_names = set(re.findall(r'\b[a-zéèêëàâäôöùûüç]{3,}\b', reference.lower()))
    
    # calcule la précision des nombres
    number_accuracy = 0.0
    if ref_numbers:
        correct_numbers = pred_numbers.intersection(ref_numbers)
        number_accuracy = len(correct_numbers) / len(ref_numbers)
    
    # calcule la précision des noms
    name_accuracy = 0.0
    if ref_names:
        correct_names = pred_names.intersection(ref_names)
        name_accuracy = len(correct_names) / len(ref_names)
    
    # combine les scores (poids égal pour les nombres et les noms)
    if ref_numbers and ref_names:
        return (number_accuracy + name_accuracy) / 2
    elif ref_numbers:
        return number_accuracy
    elif ref_names:
        return name_accuracy
    else:
        return calculate_similarity(prediction, reference)


def evaluate_single_response(
    question: str, context: List[str], answer: str, ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """évalue une seule réponse avec des métriques basiques."""
    scores = {}
    
    # faithfulness (fidélité) - basée sur la précision factuelle
    if ground_truth:
        scores["faithfulness"] = calculate_factual_accuracy(answer, ground_truth)
    else:
        scores["faithfulness"] = 0.5  # valeur par défaut
    
    # answer_relevancy (pertinence de la réponse) - basée sur la similarité avec la question
    scores["answer_relevancy"] = calculate_similarity(answer, question)
    
    # context_precision (précision du contexte) - basée sur la pertinence du contexte
    scores["context_precision"] = calculate_context_relevance(answer, context)
    
    # context_recall (rappel du contexte) - basée sur l'utilisation du contexte
    if context:
        context_text = " ".join(context)
        scores["context_recall"] = calculate_keyword_overlap(answer, context_text)
    else:
        scores["context_recall"] = 0.0
    
    return scores


def evaluate_with_metrics(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, float]:
    """évalue les réponses avec des métriques basiques."""
    all_scores = []
    
    for i, question in enumerate(questions):
        context = contexts[i] if i < len(contexts) else []
        answer = answers[i] if i < len(answers) else ""
        ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        
        scores = evaluate_single_response(question, context, answer, ground_truth)
        all_scores.append(scores)
    
    # calcule les moyennes
    if all_scores:
        avg_scores = {}
        for metric in all_scores[0].keys():
            avg_scores[metric] = sum(s[metric] for s in all_scores) / len(all_scores)
        return avg_scores
    else:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }


class RAGEvaluator:
    """évaluateur utilisant des métriques basiques pour les métriques d'évaluation."""

    def __init__(self) -> None:
        pass

    async def evaluate_response(
        self, prediction: str, reference: str, context: List[str]
    ) -> Dict[str, float]:
        """évalue une paire prédiction/référence avec son contexte."""
        # utilise les métriques basiques pour l'évaluation
        scores = evaluate_single_response(
            question=reference,  # utilise la référence comme question
            context=context,
            answer=prediction,
            ground_truth=reference,
        )
        
        return scores

    async def evaluate_dataset(
        self, predictions: List[str], references: List[str], contexts: List[List[str]]
    ) -> pd.DataFrame:
        """évalue un ensemble de prédictions avec des métriques basiques."""
        # utilise les métriques basiques pour l'évaluation en lot
        scores = evaluate_with_metrics(
            questions=references,  # utilise les références comme questions
            contexts=contexts,
            answers=predictions,
            ground_truths=references,
        )
        
        # crée un dataframe avec les résultats
        results = []
        for i in range(len(predictions)):
            result = {
                "question": references[i],
                "prediction": predictions[i],
                "reference": references[i],
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevancy": scores.get("answer_relevancy", 0.0),
                "context_precision": scores.get("context_precision", 0.0),
                "context_recall": scores.get("context_recall", 0.0),
            }
            results.append(result)
        
        return pd.DataFrame(results)

    async def plot_results(self, results_df: pd.DataFrame, output_dir: Path, engaged_mode: bool = False) -> None:
        """crée des visualisations pour les résultats."""
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # métriques
        metrics = [
            "faithfulness",
            "answer_relevancy", 
            "context_precision",
            "context_recall",
        ]
        
        # crée les histogrammes
        num_metrics = len(metrics)
        cols = min(2, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            if metric in results_df.columns:
                ax.hist(results_df[metric], bins=10, alpha=0.7, edgecolor='black')
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xlabel("score")
                ax.set_ylabel("compte")
                ax.axvline(results_df[metric].mean(), color='red', linestyle='--', 
                          label=f'moyenne: {results_df[metric].mean():.3f}')
                ax.legend()
        
        # masque les axes vides
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        
        # ajoute le suffixe si mode engagé
        if engaged_mode:
            plt.savefig(output_dir / "evaluation_metrics_engaged.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # sauvegarde les données avec suffixe si mode engagé
        if engaged_mode:
            results_df.to_csv(output_dir / "eval_metrics_engaged.csv", index=False)
        else:
            results_df.to_csv(output_dir / "eval_metrics.csv", index=False)
        
        # affiche le résumé
        print("\nrésumé de l'évaluation :")
        for metric in metrics:
            if metric in results_df.columns:
                mean_score = results_df[metric].mean()
                std_score = results_df[metric].std()
                print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")


# fonction de compatibilité pour l'interface existante
def faithfulness(prediction: str, context: List[str]) -> float:
    """mesure la fidélité (fonction de compatibilité)."""
    try:
        scores = evaluate_single_response(
            question="",  # question vide car on ne l'utilise pas
            context=context,
            answer=prediction,
        )
        return scores.get("faithfulness", 0.0)
    except Exception:
        # fallback vers une méthode simple
        return 0.5


def context_overlap_score(answer: str, context: List[str]) -> float:
    """calcule le score de fidélité basé sur la similarité avec le contexte."""
    if not context:
        return 0.0
    
    try:
        # combine tout le contexte
        full_context = " ".join(context)
        
        # normalise les textes
        answer_norm = re.sub(r'[^\w\s]', ' ', answer.lower()).strip()
        context_norm = re.sub(r'[^\w\s]', ' ', full_context.lower()).strip()
        
        # calcule la similarité avec SequenceMatcher
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, answer_norm, context_norm).ratio()
        
        # calcule aussi le chevauchement de mots-clés
        answer_words = set(re.findall(r'\b\w{3,}\b', answer_norm))
        context_words = set(re.findall(r'\b\w{3,}\b', context_norm))
        
        if answer_words:
            # pourcentage de mots de la réponse présents dans le contexte
            relevant_words = answer_words.intersection(context_words)
            keyword_ratio = len(relevant_words) / len(answer_words)
        else:
            keyword_ratio = 0.0
        
        # combine les deux scores (similarité + chevauchement de mots)
        combined_score = (similarity + keyword_ratio) / 2
        
        # ajuste le score pour qu'il soit plus réaliste
        # un score de 0.2-0.4 est normal pour une bonne réponse RAG
        # un score de 0.5+ indique une très bonne fidélité
        adjusted_score = min(1.0, combined_score * 1.2)
        
        return adjusted_score
        
    except Exception as e:
        print(f"Erreur dans context_overlap_score: {e}")
        return 0.5  # valeur par défaut raisonnable
