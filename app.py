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

from src.rag_core import RAGSystem, load_pokepedia_documents
from src.format_pokeapi_data import create_pokemon_documents

def cleanup_rag_system():
    """nettoie le système rag en cas d'erreur."""
    try:
        if "rag_system" in st.session_state:
            st.session_state.rag_system.cleanup()
    except:
        pass

# config de la page
st.set_page_config(
    page_title="Pokédex IA - Système de Questions-Réponses",
    page_icon="⚡",
    layout="wide",
)

# init de l'état
if "engaged_mode" not in st.session_state:
    st.session_state.engaged_mode = True  # mode engagé activé par défaut
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(engaged_mode=st.session_state.engaged_mode)
if "num_pokemon" not in st.session_state:
    st.session_state.num_pokemon = 0
if "num_pokepedia" not in st.session_state:
    st.session_state.num_pokepedia = 0
if "data_embedded" not in st.session_state:
    st.session_state.data_embedded = False

# chargement des données
if not st.session_state.data_embedded:
    with st.spinner("Chargement des données..."):
        try:
            # charge les documents pokeapi
            pokemon_documents = create_pokemon_documents()
            
            # charge les documents poképédia séparément pour le comptage
            pokepedia_documents = load_pokepedia_documents()
            
            # compte les documents
            pokeapi_count = len(pokemon_documents)
            pokepedia_count = len(pokepedia_documents)

            # intègre les documents
            st.info("Intégration des documents...")
            st.session_state.rag_system.embed_documents(
                pokemon_documents, pokepedia_documents
            )
            st.session_state.data_embedded = True
            st.session_state.num_pokemon = pokeapi_count
            st.session_state.num_pokepedia = pokepedia_count
            st.success(
                f"Intégration terminée ! {pokeapi_count} documents PokeAPI + {pokepedia_count} documents Poképédia chargés."
            )
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")
            st.session_state.data_embedded = False
            cleanup_rag_system()

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
    
    # mettre à jour la température du système rag
    if hasattr(st.session_state.rag_system, "llm"):
        current_temp = st.session_state.rag_system.llm.temperature
        if abs(temperature - current_temp) > 0.001:  # éviter les mises à jour inutiles
            st.session_state.rag_system.update_temperature(temperature)

    # mode engagé
    st.subheader("Mode de réponse")
    engaged_mode = st.toggle(
        "Activer le mode engagé", value=st.session_state.engaged_mode
    )
    if engaged_mode != st.session_state.engaged_mode:
        st.session_state.engaged_mode = engaged_mode
        # mettre à jour le mode du système rag existant sans réinitialiser
        if hasattr(st.session_state.rag_system, "engaged_mode"):
            st.session_state.rag_system.engaged_mode = engaged_mode
            # mettre à jour le prompt template
            st.session_state.rag_system._update_prompt_template()
    
    if engaged_mode:
        st.success("✅ Mode engagé activé - Réponses détaillées et structurées")
    else:
        st.info("ℹ️ Mode normal - Réponses concises et directes")

    # stats des données
    st.subheader("Statistiques des données")
    if st.session_state.data_embedded:
        st.write(f"Nombre de Pokémon : {st.session_state.num_pokemon}")
        st.write(f"Documents Poképédia : {st.session_state.num_pokepedia}")
        st.write("Sources : PokeAPI + Poképédia")
    else:
        st.write("Données non chargées")

    # bouton de réinitialisation
    st.subheader("Maintenance")
    if st.button("🔄 Réinitialiser l'application"):
        cleanup_rag_system()
        for key in ["data_embedded", "num_pokemon", "num_pokepedia"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

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

# vérification que les données sont chargées
if not st.session_state.data_embedded:
    st.error("❌ Les données ne sont pas encore chargées. Veuillez attendre ou rafraîchir la page.")
    st.stop()

# saisie de la question
question = st.text_input("Entrez votre question:")

if question:
    # obtention de la réponse
    with st.spinner("Génération de la réponse..."):
        try:
            result = st.session_state.rag_system.query(question)

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
                    try:
                        from src.evaluation import context_overlap_score
                        overlap = context_overlap_score(result["answer"], result["context"])
                        faithfulness = overlap
                    except Exception as e:
                        st.warning(f"Erreur lors de l'évaluation : {e}")
                        overlap = 0.5
                        faithfulness = 0.5
                
                # Affichage des indicateurs de confiance
                st.subheader("Indicateurs de Confiance")
                
                # Création de colonnes pour les métriques
                col1, col2 = st.columns(2)
                
                # Fidélité (inverse de la probabilité d'hallucination)
                hallucination_prob = 1 - faithfulness
                with col1:
                    st.metric(
                        "Probabilité d'Hallucination",
                        f"{hallucination_prob:.1%}",
                        delta=None,
                        delta_color="inverse"
                    )
                
                # Taux de recouvrement du contexte
                with col2:
                    st.metric(
                        "Recouvrement du Contexte",
                        f"{overlap:.1%}",
                        delta=None
                    )
                
                # Barre de progression pour la confiance globale
                confidence_score = faithfulness
                
                st.progress(confidence_score, text="Confiance Globale")
                
                # Avertissement si probabilité d'hallucination élevée
                if hallucination_prob > 0.3:
                    st.warning("⚠️ Attention : Cette réponse pourrait contenir des informations incorrectes ou inventées.")
                
                # Affichage du contexte (uniquement pour la recherche sémantique)
                if result.get("context") and result.get("metadata"):
                    with st.expander("Voir le Contexte Récupéré"):
                        for i, (ctx, metadata) in enumerate(
                            zip(result["context"], result["metadata"]), 1
                        ):
                            source = metadata.get("source", "unknown") if metadata else "unknown"
                            source_icon = (
                                "📊"
                                if source == "pokeapi"
                                else "📚" if source == "pokepedia" else "❓"
                            )
                            st.markdown(f"**Contexte {i}** {source_icon} ({source}):")
                            st.write(ctx)
                            st.markdown("---")
                elif result.get("context"):
                    with st.expander("Voir le Contexte Récupéré"):
                        for i, ctx in enumerate(result["context"], 1):
                            st.markdown(f"**Contexte {i}:**")
                            st.write(ctx)
                            st.markdown("---")
                        
        except ValueError as e:
            st.error(str(e))
            if "réinitialiser l'application" in str(e).lower():
                st.session_state.data_embedded = False
                cleanup_rag_system()
                st.rerun()
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            st.session_state.data_embedded = False
            cleanup_rag_system()
            st.rerun()
