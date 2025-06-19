import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.evaluation import RAGEvaluator, faithfulness

from src.rag_core import RAGSystem, load_pokepedia_documents
from src.format_pokeapi_data import create_pokemon_documents

# config de la page
st.set_page_config(
    page_title="Pokédex IA - Système de Questions-Réponses",
    page_icon="⚡",
    layout="wide",
)

# init de l'état
if "engaged_mode" not in st.session_state:
    st.session_state.engaged_mode = True  # Mode engagé activé par défaut
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(engaged_mode=st.session_state.engaged_mode)
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()
if "num_pokemon" not in st.session_state:
    st.session_state.num_pokemon = 0
if "num_pokepedia" not in st.session_state:
    st.session_state.num_pokepedia = 0

# chargement des données
if "data_embedded" not in st.session_state:
    with st.spinner("Chargement des données..."):
        try:
            # charge les documents PokeAPI
            pokemon_documents = create_pokemon_documents()
            
            # charge les documents Poképédia séparément pour le comptage
            pokepedia_documents = load_pokepedia_documents()
            
            # compte les documents
            pokeapi_count = len(pokemon_documents)
            pokepedia_count = len(pokepedia_documents)

            # intègre les documents
            st.info("Intégration des documents...")
            st.session_state.rag_system.embed_documents(pokemon_documents, pokepedia_documents)
            st.session_state.data_embedded = True
            st.session_state.num_pokemon = pokeapi_count
            st.session_state.num_pokepedia = pokepedia_count
            st.success(
                f"Intégration terminée ! {pokeapi_count} documents PokeAPI + {pokepedia_count} documents Poképédia chargés."
            )
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")
            st.session_state.data_embedded = False

# titre et description
st.title("⚡ Pokédex IA - Système de Questions-Réponses")
st.markdown(
    """
Cette application utilise un système RAG (Retrieval-Augmented Generation) pour répondre à vos questions
sur les Pokémon. Le système utilise le modèle Gemini de Google pour la génération
et ChromaDB pour le stockage et la récupération des informations.

Les données proviennent de deux sources principales :
- **PokeAPI** : Informations détaillées sur chaque Pokémon (statistiques, types, capacités, descriptions officielles)
- **Poképédia** : Contenu enrichi en français avec descriptions détaillées, biologie, comportement, habitat, mythologie et faits divers

Le système utilise une recherche vectorielle avancée avec :
- Métadonnées enrichies incluant les informations d'index (types, statuts, habitats, couleurs)
- Intégration automatique des données Poképédia pour des réponses plus riches et détaillées
- Recherche sémantique pour comprendre le contexte et l'intention des questions
"""
)

# barre latérale
with st.sidebar:
    st.header("Paramètres")

    # paramètres du modèle
    st.subheader("Paramètres du modèle")
    temperature = st.slider("Température", 0.0, 1.0, 0.0, 0.1)
    
    # Mettre à jour la température du système RAG
    if hasattr(st.session_state.rag_system, 'llm'):
        current_temp = st.session_state.rag_system.llm.temperature
        if abs(temperature - current_temp) > 0.001:  # Éviter les mises à jour inutiles
            st.session_state.rag_system.update_temperature(temperature)

    # mode engagé
    st.subheader("Mode de réponse")
    engaged_mode = st.toggle("Activer le mode engagé", value=st.session_state.engaged_mode)
    if engaged_mode != st.session_state.engaged_mode:
        st.session_state.engaged_mode = engaged_mode
        # Mettre à jour le mode du système RAG existant sans réinitialiser
        if hasattr(st.session_state.rag_system, 'engaged_mode'):
            st.session_state.rag_system.engaged_mode = engaged_mode
            # Mettre à jour le prompt template
            st.session_state.rag_system._update_prompt_template()
    
    if engaged_mode:
        st.success("✅ Mode engagé activé - Réponses détaillées et structurées")
    else:
        st.info("ℹ️ Mode normal - Réponses concises et directes")

    # stats des données
    st.subheader("Statistiques des données")
    if "data_embedded" in st.session_state and st.session_state.data_embedded:
        st.write(f"Nombre de Pokémon : {st.session_state.num_pokemon}")
        st.write(f"Documents Poképédia : {st.session_state.num_pokepedia}")
        st.write("Sources : PokeAPI + Poképédia")
    else:
        st.write("Données non chargées")

    # exemples de questions
    st.subheader("Exemples de questions")
    st.markdown(
        """
    - Liste les Pokémon légendaires
    - Quels sont les Pokémon mythiques ?
    - Décris-moi Pikachu
    - Quelles sont les stats de base de Charizard ?
    - Qui a le plus d'attaque entre Lapras et Rattata ?
    - Raconte-moi l'histoire et la mythologie de Mewtwo
    - Décris le comportement et l'habitat de Bulbizarre
    - Quels sont les faits intéressants sur Arcanin ?
    """
    )

# contenu principal
st.header("Posez votre question")

# saisie de la question
question = st.text_input("Entrez votre question:")

if question:
    # obtention de la réponse
    with st.spinner("Génération de la réponse..."):
        try:
            result = st.session_state.rag_system.query(question)

            # affichage de la réponse
            st.info("Recherche sémantique (vecteurs)")
            st.subheader("Réponse")
            st.write(result["answer"])

            # évaluation de la réponse
            with st.spinner("Évaluation de la réponse..."):
                overlap = faithfulness(result["answer"], result["context"])
                faithfulness_score = overlap

            # indicateurs de confiance
            st.subheader("Indicateurs de confiance")

            # colonnes pour les métriques
            col1, col2 = st.columns(2)

            # fidélité
            hallucination_prob = 1 - faithfulness_score
            with col1:
                st.metric(
                    "Probabilité d'hallucination",
                    f"{hallucination_prob:.1%}",
                    delta=None,
                    delta_color="inverse",
                )

            # recouvrement du contexte
            with col2:
                st.metric("Recouvrement du contexte", f"{overlap:.1%}", delta=None)

            # barre de confiance
            confidence_score = faithfulness_score
            st.progress(confidence_score, text="Confiance globale")

            # avertissement si hallucination élevée
            if hallucination_prob > 0.3:
                st.warning("⚠️ Attention : réponse potentiellement incorrecte")

            # affichage du contexte
            if result["context"]:
                with st.expander("Voir le contexte"):
                    for i, (ctx, metadata) in enumerate(zip(result["context"], result["metadata"]), 1):
                        source = metadata.get("source", "unknown")
                        source_icon = "📊" if source == "pokeapi" else "📚" if source == "pokepedia" else "❓"
                        st.markdown(f"**Contexte {i}** {source_icon} ({source}):")
                        st.write(ctx)
                        st.markdown("---")
        except ValueError as e:
            st.error(str(e))
            if "réinitialiser l'application" in str(e).lower():
                st.session_state.data_embedded = False
                st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.data_embedded = False
            st.rerun()

# section d'évaluation
st.header("Évaluation")
with st.expander("Ajouter une référence"):
    reference = st.text_area("Entrez la réponse de référence:")
    if st.button("Sauvegarder la référence"):
        st.session_state.reference_answer = reference
        st.success("Référence sauvegardée!")

# journal des hallucinations
if "answer" in locals() and "reference_answer" in st.session_state:
    if faithfulness_score < 0.7:  # seuil d'hallucination
        log_path = Path("hallucinations.csv")
        log_df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(),
                    "question": question,
                    "prediction": result["answer"],
                    "reference": st.session_state.reference_answer,
                    "faithfulness_score": faithfulness_score,
                    "search_type": "semantic",
                }
            ]
        )

        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
