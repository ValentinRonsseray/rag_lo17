import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.evaluation import RAGEvaluator, context_overlap_score

from src.rag_core import RAGSystem
from src.format_pokeapi_data import create_pokemon_documents

# config de la page
st.set_page_config(
    page_title="Pokédex IA - Système de Questions-Réponses",
    page_icon="⚡",
    layout="wide",
)

# init de l'état
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()

# chargement des données
if "data_embedded" not in st.session_state:
    with st.spinner("chargement des données..."):
        try:
            # charge les documents
            pokemon_documents = create_pokemon_documents()

            # intègre les documents
            st.info("intégration des documents...")
            st.session_state.rag_system.embed_documents(pokemon_documents)
            st.session_state.data_embedded = True
            st.session_state.num_pokemon = len(pokemon_documents)
            st.success(
                f"intégration terminée ! {len(pokemon_documents)} documents chargés."
            )
        except Exception as e:
            st.error(f"erreur de chargement : {e}")
            st.session_state.data_embedded = False

# titre et description
st.title("⚡ Pokédex IA - Système de Questions-Réponses")
st.markdown(
    """
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
"""
)

# barre latérale
with st.sidebar:
    st.header("paramètres")

    # paramètres du modèle
    st.subheader("paramètres du modèle")
    temperature = st.slider("température", 0.0, 1.0, 0.0, 0.1)

    # stats des données
    st.subheader("stats des données")
    if "data_embedded" in st.session_state:
        st.write(f"nombre de pokémon : {st.session_state.num_pokemon}")
        st.write("sources : pokeapi")

    # exemples de questions
    st.subheader("exemples de questions")
    st.markdown(
        """
    - quels sont les pokémon de type feu ?
    - liste les pokémon légendaires
    - quels sont les pokémon mythiques ?
    - décris-moi pikachu
    - quelles sont les stats de base de charizard ?
    """
    )

# contenu principal
st.header("posez votre question")

# saisie de la question
question = st.text_input("entrez votre question:")

if question:
    # obtention de la réponse
    with st.spinner("génération de la réponse..."):
        try:
            result = st.session_state.rag_system.query(question)

            # affichage du type de recherche
            search_type = result.get("search_type", "semantic")
            if search_type == "exact":
                st.success("recherche exacte (index inverse)")
                # pas de métriques pour les recherches exactes
                st.subheader("réponse")
                st.write(result["answer"])
            else:
                st.info("recherche sémantique (vecteurs)")
                # affichage de la réponse
                st.subheader("réponse")
                st.write(result["answer"])

                # évaluation de la réponse
                with st.spinner("évaluation de la réponse..."):
                    overlap = context_overlap_score(result["answer"], result["context"])
                    faithfulness = overlap

                # indicateurs de confiance
                st.subheader("indicateurs de confiance")

                # colonnes pour les métriques
                col1, col2 = st.columns(2)

                # fidélité
                hallucination_prob = 1 - faithfulness
                with col1:
                    st.metric(
                        "probabilité d'hallucination",
                        f"{hallucination_prob:.1%}",
                        delta=None,
                        delta_color="inverse",
                    )

                # recouvrement du contexte
                with col2:
                    st.metric("recouvrement du contexte", f"{overlap:.1%}", delta=None)

                # barre de confiance
                confidence_score = faithfulness
                st.progress(confidence_score, text="confiance globale")

                # avertissement si hallucination élevée
                if hallucination_prob > 0.3:
                    st.warning("⚠️ attention : réponse potentiellement incorrecte")

                # affichage du contexte
                if result["context"]:
                    with st.expander("voir le contexte"):
                        for i, ctx in enumerate(result["context"], 1):
                            st.markdown(f"**contexte {i}:**")
                            st.write(ctx)
                            st.markdown("---")
        except ValueError as e:
            st.error(str(e))
            if "réinitialiser l'application" in str(e).lower():
                st.session_state.data_embedded = False
                st.rerun()
        except Exception as e:
            st.error(f"erreur : {e}")
            st.session_state.data_embedded = False
            st.rerun()

# section d'évaluation
st.header("évaluation")
with st.expander("ajouter une référence"):
    reference = st.text_area("entrez la réponse de référence:")
    if st.button("sauvegarder la référence"):
        st.session_state.reference_answer = reference
        st.success("référence sauvegardée!")

# journal des hallucinations
if "answer" in locals() and "reference_answer" in st.session_state:
    if faithfulness < 0.7:  # seuil d'hallucination
        log_path = Path("hallucinations.csv")
        log_df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(),
                    "question": question,
                    "prediction": result["answer"],
                    "reference": st.session_state.reference_answer,
                    "faithfulness_score": faithfulness,
                    "search_type": search_type,
                }
            ]
        )

        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
