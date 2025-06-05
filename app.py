"""
Application Streamlit pour le système RAG Pokémon.
"""

import os
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
from langchain.docstore.document import Document

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator

# Initialisation de l'état de la session
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()

# Chargement et intégration des données Pokémon
if "pokemon_data_embedded" not in st.session_state:
    # Lecture des données Pokémon
    pokemon_df = pd.read_csv("data/pokemon_basic_stats.csv")
    
    # Conversion de chaque ligne Pokémon en document
    documents = []
    for _, row in pokemon_df.iterrows():
        # Création d'une description textuelle du Pokémon
        text = f"Le Pokémon {row['name']} (ID: {row['id']}) est de type {row['types']}. "
        text += f"Il possède les capacités suivantes : {row['abilities']}. "
        text += f"Ses statistiques de base sont : PV: {row['hp']}, Attaque: {row['attack']}, Défense: {row['defense']}, "
        text += f"Attaque Spéciale: {row['special-attack']}, Défense Spéciale: {row['special-defense']}, Vitesse: {row['speed']}. "
        text += f"Il pèse {row['weight']} unités et mesure {row['height']} unités de hauteur."
        
        # Création du document avec métadonnées
        doc = Document(
            page_content=text,
            metadata={
                "id": row["id"],
                "name": row["name"],
                "types": row["types"],
                "abilities": row["abilities"]
            }
        )
        documents.append(doc)
    
    # Intégration des documents
    st.session_state.rag_system.embed_documents(documents)
    st.session_state.pokemon_data_embedded = True

# Configuration de la page
st.set_page_config(
    page_title="Pokédex IA - Système de Questions-Réponses",
    page_icon="⚡",
    layout="wide"
)

# Titre et description
st.title("⚡ Pokédex IA - Système de Questions-Réponses")
st.markdown("""
Cette application utilise un système RAG (Retrieval-Augmented Generation) pour répondre à vos questions
sur les Pokémon. Le système utilise le modèle Gemini de Google pour la génération
et ChromaDB pour le stockage et la récupération des informations.
""")

# Barre latérale
with st.sidebar:
    st.header("Paramètres")
    
    # Paramètres du modèle
    st.subheader("Paramètres du Modèle")
    temperature = st.slider("Température", 0.0, 1.0, 0.0, 0.1)
    
    # Téléchargement de documents
    st.subheader("Téléchargement de Documents")
    uploaded_file = st.file_uploader("Télécharger un document", type=["txt", "pdf"])
    
    if uploaded_file:
        # Sauvegarde du fichier téléchargé
        save_path = Path("data/uploads") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Chargement et intégration du document
        with st.spinner("Traitement du document..."):
            docs = st.session_state.rag_system.load_documents([str(save_path)])
            st.session_state.rag_system.embed_documents(docs)
            st.success("Document traité avec succès!")

# Contenu principal
st.header("Posez votre Question")

# Saisie de la question
question = st.text_input("Entrez votre question:")

if question:
    # Obtention de la réponse
    with st.spinner("Génération de la réponse..."):
        result = st.session_state.rag_system.query(question)
        
        # Affichage de la réponse
        st.subheader("Réponse")
        st.write(result["answer"])
        
        # Affichage du contexte
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
            "faithfulness_score": scores["faithfulness"]
        }])
        
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False) 