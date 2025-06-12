"""
Application Streamlit pour le système RAG Pokémon.
"""

import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator
from src.format_pokeapi_data import create_pokemon_documents

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
        else:
            st.info("Recherche sémantique (vecteurs)")
        
        # Affichage de la réponse
        st.subheader("Réponse")
        st.write(result["answer"])
        
        # Affichage du contexte (uniquement pour la recherche sémantique)
        if search_type == "semantic" and result["context"]:
            with st.expander("Voir le Contexte Récupéré"):
                for i, ctx in enumerate(result["context"], 1):
                    st.markdown(f"**Contexte {i}:**")
                    st.write(ctx)
                    st.markdown("---")
        
        # Évaluation de la réponse
        if "reference_answer" in st.session_state:
            scores = st.session_state.evaluator.evaluate_response(
                result["answer"],
                st.session_state.reference_answer,
                result["context"]
            )
            
            # Affichage des scores
            st.subheader("Scores d'Évaluation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correspondance Exacte", f"{scores['exact_match']:.2f}")
            with col2:
                st.metric("Score F1", f"{scores['f1_score']:.2f}")
            with col3:
                st.metric("Fidélité", f"{scores['faithfulness']:.2f}")

# Section d'évaluation
st.header("Évaluation")
with st.expander("Ajouter une Réponse de Référence"):
    reference = st.text_area("Entrez la réponse de référence:")
    if st.button("Sauvegarder la Référence"):
        st.session_state.reference_answer = reference
        st.success("Réponse de référence sauvegardée!")

# Journal des hallucinations
if "answer" in locals() and "reference_answer" in st.session_state:
    if scores["faithfulness"] < 0.7:  # Seuil pour l'hallucination
        log_path = Path("hallucinations.csv")
        log_df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "question": question,
            "prediction": result["answer"],
            "reference": st.session_state.reference_answer,
            "faithfulness_score": scores["faithfulness"],
            "search_type": search_type
        }])
        
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False) 