import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd

# fix pour le problème de protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# fix pour le problème de distutils en python 3.12+
try:
    import distutils
except ImportError:
    import setuptools._distutils as distutils

from src.evaluation import RAGEvaluator, faithfulness

from src.rag_core import RAGSystem, load_pokepedia_documents
from src.format_pokeapi_data import create_pokemon_documents

# config de la page
st.set_page_config(
    page_title="pokédex ia - système de questions-réponses",
    page_icon="⚡",
    layout="wide",
)

# init de l'état
if "engaged_mode" not in st.session_state:
    st.session_state.engaged_mode = True  # mode engagé activé par défaut
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
    with st.spinner("chargement des données..."):
        try:
            # charge les documents pokeapi
            pokemon_documents = create_pokemon_documents()
            
            # charge les documents poképédia séparément pour le comptage
            pokepedia_documents = load_pokepedia_documents()
            
            # compte les documents
            pokeapi_count = len(pokemon_documents)
            pokepedia_count = len(pokepedia_documents)

            # intègre les documents
            st.info("intégration des documents...")
            st.session_state.rag_system.embed_documents(
                pokemon_documents, pokepedia_documents
            )
            st.session_state.data_embedded = True
            st.session_state.num_pokemon = pokeapi_count
            st.session_state.num_pokepedia = pokepedia_count
            st.success(
                f"intégration terminée ! {pokeapi_count} documents pokeapi + {pokepedia_count} documents poképédia chargés."
            )
        except Exception as e:
            st.error(f"erreur de chargement : {e}")
            st.session_state.data_embedded = False

# titre et description
st.title("⚡ pokédex ia - système de questions-réponses")
st.markdown(
    """
cette application utilise un système rag (retrieval-augmented generation) pour répondre à vos questions
sur les pokémon. le système utilise le modèle gemini de google pour la génération
et chromadb pour le stockage et la récupération des informations.

les données proviennent de deux sources principales :
- **pokeapi** : informations détaillées sur chaque pokémon (statistiques, types, capacités, descriptions officielles)
- **poképédia** : contenu enrichi en français avec descriptions détaillées, biologie, comportement, habitat, mythologie et faits divers

le système utilise une recherche vectorielle avancée avec :
- métadonnées enrichies incluant les informations d'index (types, statuts, habitats, couleurs)
- intégration automatique des données poképédia pour des réponses plus riches et détaillées
- recherche sémantique pour comprendre le contexte et l'intention des questions

**évaluation ragas** : le système utilise ragas (retrieval-augmented generation assessment) pour évaluer la qualité des réponses.
"""
)

# barre latérale
with st.sidebar:
    st.header("paramètres")

    # paramètres du modèle
    st.subheader("paramètres du modèle")
    temperature = st.slider("température", 0.0, 1.0, 0.0, 0.1)
    
    # mettre à jour la température du système rag
    if hasattr(st.session_state.rag_system, "llm"):
        current_temp = st.session_state.rag_system.llm.temperature
        if abs(temperature - current_temp) > 0.001:  # éviter les mises à jour inutiles
            st.session_state.rag_system.update_temperature(temperature)

    # mode engagé
    st.subheader("mode de réponse")
    engaged_mode = st.toggle(
        "activer le mode engagé", value=st.session_state.engaged_mode
    )
    if engaged_mode != st.session_state.engaged_mode:
        st.session_state.engaged_mode = engaged_mode
        # mettre à jour le mode du système rag existant sans réinitialiser
        if hasattr(st.session_state.rag_system, "engaged_mode"):
            st.session_state.rag_system.engaged_mode = engaged_mode
            # mettre à jour le prompt template
            st.session_state.rag_system._update_prompt_template()
    
    if engaged_mode:
        st.success("✅ mode engagé activé - réponses détaillées et structurées")
    else:
        st.info("ℹ️ mode normal - réponses concises et directes")

    # stats des données
    st.subheader("statistiques des données")
    if "data_embedded" in st.session_state and st.session_state.data_embedded:
        st.write(f"nombre de pokémon : {st.session_state.num_pokemon}")
        st.write(f"documents poképédia : {st.session_state.num_pokepedia}")
        st.write("sources : pokeapi + poképédia")
    else:
        st.write("données non chargées")

    # exemples de questions
    st.subheader("exemples de questions")
    st.markdown(
        """
    - liste les pokémon légendaires
    - quels sont les pokémon mythiques ?
    - décris-moi pikachu
    - quelles sont les stats de base de charizard ?
    - qui a le plus d'attaque entre lapras et rattata ?
    - raconte-moi l'histoire et la mythologie de mewtwo
    - décris le comportement et l'habitat de bulbizarre
    - quels sont les faits intéressants sur arcanin ?
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

            # affichage de la réponse
            st.info("recherche sémantique (vecteurs)")
            st.subheader("réponse")
            st.write(result["answer"])

            # évaluation de la réponse avec ragas
            with st.spinner("évaluation ragas de la réponse..."):
                try:
                    # utilise ragas pour l'évaluation
                    from src.evaluation import evaluate_single_response
                    ragas_scores = evaluate_single_response(
                        question=question,
                        context=result["context"],
                        answer=result["answer"]
                    )
                    
                    faithfulness_score = ragas_scores.get("faithfulness", 0.0)
                    answer_relevancy = ragas_scores.get("answer_relevancy", 0.0)
                    context_precision = ragas_scores.get("context_precision", 0.0)
                    context_recall = ragas_scores.get("context_recall", 0.0)
                    
                except Exception as e:
                    st.warning(f"erreur lors de l'évaluation ragas : {e}")
                    # fallback vers l'ancienne méthode
                    faithfulness_score = faithfulness(result["answer"], result["context"])
                    answer_relevancy = 0.5
                    context_precision = 0.5
                    context_recall = 0.5

            # indicateurs de confiance ragas
            st.subheader("métriques ragas")

            # colonnes pour les métriques
            col1, col2 = st.columns(2)

            # faithfulness (fidélité)
            with col1:
                st.metric(
                    "faithfulness",
                    f"{faithfulness_score:.3f}",
                    delta=None,
                )

            # answer_relevancy (pertinence de la réponse)
            with col2:
                st.metric(
                    "answer_relevancy", 
                    f"{answer_relevancy:.3f}",
                    delta=None,
                )

            # context_precision et context_recall
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric(
                    "context_precision",
                    f"{context_precision:.3f}",
                    delta=None,
                )
            
            with col4:
                st.metric(
                    "context_recall",
                    f"{context_recall:.3f}",
                    delta=None,
                )

            # barre de confiance globale (moyenne des métriques ragas)
            confidence_score = (faithfulness_score + answer_relevancy + context_precision + context_recall) / 4
            st.progress(confidence_score, text="confiance globale (moyenne ragas)")

            # avertissement si scores faibles
            if faithfulness_score < 0.7:
                st.warning("⚠️ attention : faible faithfulness - réponse potentiellement incorrecte")
            
            if answer_relevancy < 0.5:
                st.warning("⚠️ attention : faible answer_relevancy - réponse potentiellement hors sujet")

            # affichage du contexte
            if result["context"]:
                with st.expander("voir le contexte"):
                    for i, (ctx, metadata) in enumerate(
                        zip(result["context"], result["metadata"]), 1
                    ):
                        source = metadata.get("source", "unknown")
                        source_icon = (
                            "📊"
                            if source == "pokeapi"
                            else "📚" if source == "pokepedia" else "❓"
                        )
                        st.markdown(f"**contexte {i}** {source_icon} ({source}):")
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
