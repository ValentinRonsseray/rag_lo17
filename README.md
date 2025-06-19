# RAG Pokémon Question Answering System

Un système de Questions-Réponses basé sur RAG (Retrieval-Augmented Generation) spécialisé dans l'univers Pokémon, utilisant LangChain, le modèle Gemini de Google et ChromaDB.

## Fonctionnalités

- **Sources de données multiples** : PokeAPI (statistiques, types, capacités) + Poképédia (descriptions, biologie, comportement)
- **Recherche vectorielle avancée** avec métadonnées enrichies (types, statuts, habitats, couleurs)
- **Mode de réponse configurable** : Normal (concis) ou Engagé (détaillé)
- **Évaluation complète** avec métriques inspirées de RAGAS
- **Détection d'hallucinations** et journalisation automatique
- **Interface web Streamlit** intuitive
- **Scraping automatique** des données Poképédia
- **Index hybrides** pour une recherche optimisée
- **Support Docker** pour le déploiement

## Architecture

### Sources de données
- **PokeAPI** : Données techniques officielles (statistiques, types, capacités, évolutions)
- **Poképédia** : Contenu enrichi en français (descriptions, biologie, comportement, habitat, mythologie)
- **Index métadonnées** : Types, statuts légendaires, habitats, couleurs

### Système RAG
- **Embeddings** : Google Generative AI (models/embedding-001)
- **LLM** : Gemini 2.0 Flash
- **Vector Store** : ChromaDB avec métadonnées enrichies
- **Recherche** : Vectorielle sémantique avec filtrage par métadonnées

## Prérequis

- Python 3.12 ou supérieur
- Clé API Google pour le modèle Gemini
- Docker (optionnel, pour le déploiement conteneurisé)

## Installation

1. **Cloner le repository** :
```bash
git clone <repository-url>
cd rag-lo17
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Configurer la clé API Google** :
```bash
# Créer un fichier .env
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

## Utilisation

### Lancement local

1. **Lancer l'application** :
```bash
python main.py --run
```

Le script vérifie automatiquement la présence des données et les télécharge si nécessaire :
- **PokeAPI** : Données Pokémon de la génération 1
- **Poképédia** : Scraping automatique des pages contenant "est un pokemon"
- **Index** : Construction automatique des index par type, statut, habitat, couleur

2. **Ouvrir l'interface** : `http://localhost:8501`

### Utilisation avec Docker

1. **Construire l'image** :
```bash
docker build -t rag-pokemon .
```

2. **Lancer le conteneur** :
```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY='your-api-key' rag-pokemon
```

## Structure du projet

```
rag-lo17/
├── app.py                      # Application Streamlit
├── main.py                     # Script principal de lancement
├── src/
│   ├── rag_core.py            # Système RAG principal
│   ├── evaluation.py          # Métriques d'évaluation
│   ├── evaluate_rag.py        # Script d'évaluation complet
│   ├── scrap_pokeapi.py       # Téléchargement PokeAPI
│   ├── scrap_pokepedia.py     # Scraping Poképédia
│   ├── build_pokemon_index.py # Construction des index
│   ├── format_pokeapi_data.py # Formatage des données
│   └── pokepedia_data.py      # Gestion des données Poképédia
├── data/
│   ├── pokeapi/               # Données PokeAPI
│   ├── pokepedia/             # Données Poképédia
│   └── indexes/               # Index de recherche
├── evaluation_results/        # Résultats d'évaluation
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Configuration Docker
└── README.md                  # Documentation
```

## Fonctionnalités avancées

### Mode de réponse
- **Mode Normal** : Réponses concises (3-5 phrases)
- **Mode Engagé** : Réponses détaillées et structurées

### Types de questions supportées
- **Statistiques** : "Quelles sont les stats de Pikachu ?"
- **Descriptions** : "Décris le comportement de Charizard"
- **Catégorisation** : "Quels sont les Pokémon de type feu ?"
- **Comparaisons** : "Qui est plus rapide entre Mew et Machopeur ?"
- **Lore et mythologie** : "Raconte l'histoire de Mewtwo"

### Métadonnées enrichies
- **Types** : pokemon_types (ex: "fire, flying")
- **Statuts** : is_legendary, is_mythical, is_baby
- **Habitat** : habitat (ex: "mountain", "forest")
- **Couleur** : color (ex: "red", "blue")

## Évaluation

Le système inclut une évaluation complète avec métriques inspirées de RAGAS :

### Métriques calculées
- **F1 Score** : Précision et rappel de la réponse
- **Similarité** : Similarité sémantique avec la référence
- **Précision contexte** : Pertinence du contexte récupéré
- **Rappel contexte** : Complétude du contexte par rapport à la référence
- **Fidélité** : Proportion de la réponse basée sur le contexte

### Lancement de l'évaluation
```bash
# Utiliser le jeu de questions par défaut
python src/evaluate_rag.py

# Utiliser un jeu de questions personnalisé
python src/evaluate_rag.py path/to/questions.json
```

### Résultats
- **Rapport détaillé** : `evaluation_results/evaluation_report.txt`
- **Données brutes** : `evaluation_results/evaluation_results.csv`
- **Visualisations** : `evaluation_results/evaluation_metrics.png`

## Détection d'hallucinations

Le système détecte automatiquement les hallucinations potentielles :
- **Seuil** : Fidélité < 0.7
- **Journalisation** : `hallucinations.csv`
- **Indicateurs visuels** : Barres de confiance dans l'interface

## Configuration

### Variables d'environnement
```bash
GOOGLE_API_KEY=your-api-key-here
```

### Paramètres du modèle
- **Température** : 0.0 (déterministe) à 1.0 (créatif)
- **Max tokens** : 256 (normal) à 512 (engagé)
- **Documents récupérés** : 2 (normal) à 4 (engagé)

## Améliorations récentes

- ✅ **Intégration Poképédia** : Contenu enrichi en français
- ✅ **Métadonnées enrichies** : Index par type, statut, habitat, couleur
- ✅ **Prompts optimisés** : Instructions précises pour l'utilisation du contexte
- ✅ **Évaluation complète** : Rapport détaillé avec statistiques
- ✅ **Interface améliorée** : Mode engagé par défaut, indicateurs de confiance
- ✅ **Correction protobuf** : Compatibilité avec les nouvelles versions

## Contribution

1. Fork le repository
2. Créer une branche feature
3. Commiter les changements
4. Pousser vers la branche
5. Créer une Pull Request

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
