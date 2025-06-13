"""
Application Streamlit pour le système RAG Pokémon.
"""

import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncio

# Utility to safely run async functions in a synchronous context
def run_async_task(coro):
    """Run an async task, even if an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
from langchain.docstore.document import Document

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator
from src.format_pokeapi_data import create_pokemon_documents
from ragas import SingleTurnSample

# Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Pokédex IA - Système de Questions-Réponses",
    page_icon="⚡",
    layout="wide"
)

# Initialisation de l'état de la session
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()

# Chargement et intégration des données
if "data_embedded" not in st.session_state:
    with st.spinner("Chargement des données Pokémon depuis PokeAPI..."):
        # Chargement des documents Pokémon depuis PokeAPI
        pokemon_documents = create_pokemon_documents()
        
        # Intégration des documents
        st.info("Intégration des documents dans le système RAG...")
        st.session_state.rag_system.embed_documents(pokemon_documents)
        st.session_state.data_embedded = True
        st.session_state.num_pokemon = len(pokemon_documents)
        st.success(f"Intégration terminée ! {len(pokemon_documents)} documents chargés.")

# Titre et description
st.title("⚡ Pokédex IA - Système de Questions-Réponses")
st.markdown("""
Cette application utilise un système RAG (Retrieval-Augmented Generation) pour répondre à vos questions
sur les Pokémon. Le système utilise le modèle Gemini de Google pour la génération
et ChromaDB pour le stockage et la récupération des informations.

Les données proviennent directement de l'API Pokémon officielle (PokeAPI) et incluent :
- Informations détaillées sur chaque Pokémon
- Statistiques de base
- Types et capacités
- Descriptions en français
- Formes alternatives (Méga-évolutions, formes régionales, etc.)

Le système utilise un index hybride qui combine :
- Recherche vectorielle pour les questions complexes
- Index inverses pour les recherches exactes (types, statuts, etc.)
""")

# Barre latérale
with st.sidebar:
    st.header("Paramètres")
    
    # Paramètres du modèle
    st.subheader("Paramètres du Modèle")
    temperature = st.slider("Température", 0.0, 1.0, 0.0, 0.1)
    
    # Statistiques des données
    st.subheader("Statistiques des Données")
    if "data_embedded" in st.session_state:
        st.write(f"Nombre de Pokémon : {st.session_state.num_pokemon}")
        st.write("Sources : PokeAPI")
    
    # Exemples de questions
    st.subheader("Exemples de Questions")
    st.markdown("""
    - Quels sont les Pokémon de type feu ?
    - Liste les Pokémon légendaires
    - Quels sont les Pokémon mythiques ?
    - Décris-moi Pikachu
    - Quelles sont les statistiques de base de Charizard ?
    """)

# Contenu principal
st.header("Posez votre Question")

# Saisie de la question
question = st.text_input("Entrez votre question:")

if question:
    # Obtention de la réponse
    with st.spinner("Génération de la réponse..."):
        result = st.session_state.rag_system.query(question)
        
        # Affichage du type de recherche
        search_type = result.get("search_type", "semantic")
        if search_type == "exact":
            st.success("Recherche exacte (index inverse)")
            # Pour les recherches exactes, on n'affiche pas les métriques de confiance
            st.subheader("Réponse")
            st.write(result["answer"])
        else:
            st.info("Recherche sémantique (vecteurs)")
            # Affichage de la réponse
            st.subheader("Réponse")
            st.write(result["answer"])
            
            # Évaluation de la réponse
            with st.spinner("Évaluation de la réponse..."):
                # Convertir le contexte en objets Document
                context_docs = [Document(page_content=ctx) for ctx in result["context"]]
                
                # Créer un SingleTurnSample pour les métriques RAGAS
                sample = SingleTurnSample(
                    prompt=question,
                    response=result["answer"],
                    reference="",  # Pas de référence pour l'évaluation en temps réel
                    context=context_docs,
                )
                
                # Calculer uniquement les métriques qui ne nécessitent pas de référence
                try:
                    response_relevancy = run_async_task(
                        st.session_state.evaluator.response_relevancy.single_turn_ascore(sample)
                    )
                    context_precision = run_async_task(
                        st.session_state.evaluator.context_precision.single_turn_ascore(sample)
                    )
                    context_recall = run_async_task(
                        st.session_state.evaluator.context_recall.single_turn_ascore(sample)
                    )
                    faithfulness = run_async_task(
                        st.session_state.evaluator.faithfulness_metric.single_turn_ascore(sample)
                    )
                except Exception as e:
                    print(f"Erreur lors du calcul des métriques: {e}")
                    response_relevancy = 0.0
                    context_precision = 0.0
                    context_recall = 0.0
                    faithfulness = 0.0
            
            # Affichage des indicateurs de confiance
            st.subheader("Indicateurs de Confiance")
            
            # Création de colonnes pour les métriques
            col1, col2, col3, col4 = st.columns(4)
            
            # Fidélité (inverse de la probabilité d'hallucination)
            hallucination_prob = 1 - faithfulness
            with col1:
                st.metric(
                    "Probabilité d'Hallucination",
                    f"{hallucination_prob:.1%}",
                    delta=None,
                    delta_color="inverse"
                )
            
            # Pertinence de la réponse
            with col2:
                st.metric(
                    "Pertinence",
                    f"{response_relevancy:.1%}",
                    delta=None
                )
            
            # Précision du contexte
            with col3:
                st.metric(
                    "Précision du Contexte",
                    f"{context_precision:.1%}",
                    delta=None
                )
            
            # Rappel du contexte
            with col4:
                st.metric(
                    "Rappel du Contexte",
                    f"{context_recall:.1%}",
                    delta=None
                )
            
            # Barre de progression pour la confiance globale
            confidence_score = (
                faithfulness * 0.4 +  # Poids plus important pour la fidélité
                response_relevancy * 0.3 +
                context_precision * 0.15 +
                context_recall * 0.15
            )
            
            st.progress(confidence_score, text="Confiance Globale")
            
            # Avertissement si probabilité d'hallucination élevée
            if hallucination_prob > 0.3:
                st.warning("⚠️ Attention : Cette réponse pourrait contenir des informations incorrectes ou inventées.")
            
            # Affichage du contexte (uniquement pour la recherche sémantique)
            if result["context"]:
                with st.expander("Voir le Contexte Récupéré"):
                    for i, ctx in enumerate(result["context"], 1):
                        st.markdown(f"**Contexte {i}:**")
                        st.write(ctx)
                        st.markdown("---")

# Section d'évaluation
st.header("Évaluation")
with st.expander("Ajouter une Réponse de Référence"):
    reference = st.text_area("Entrez la réponse de référence:")
    if st.button("Sauvegarder la Référence"):
        st.session_state.reference_answer = reference
        st.success("Réponse de référence sauvegardée!")

# Journal des hallucinations
if "answer" in locals() and "reference_answer" in st.session_state:
    if faithfulness < 0.7:  # Seuil pour l'hallucination
        log_path = Path("hallucinations.csv")
        log_df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "question": question,
            "prediction": result["answer"],
            "reference": st.session_state.reference_answer,
            "faithfulness_score": faithfulness,
            "search_type": search_type
        }])
        
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False) 