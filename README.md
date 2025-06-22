# RAG Pokémon Question Answering System

Un système de questions-réponses basé sur RAG (Retrieval-Augmented Generation) spécialisé dans l'univers Pokémon, utilisant LangChain, le modèle Gemini de Google et ChromaDB.

## Fonctionnalités

- **Sources de données multiples** : PokeAPI (statistiques, types, capacités) + Poképédia (descriptions, biologie, comportement)
- **Recherche vectorielle avancée** avec métadonnées enrichies (types, statuts, habitats, couleurs)
- **Mode de réponse configurable** : Normal (concis) ou Engagé (détaillé)
- **Évaluation simplifiée** : Indicateurs de confiance et détection d'hallucinations
- **Interface web Streamlit** intuitive
- **Scraping automatique** des données Poképédia
- **Index hybrides** pour une recherche optimisée

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

### Évaluation simplifiée
- **Probabilité d'hallucination** : Basée sur la fidélité au contexte
- **Recouvrement du contexte** : Pourcentage de mots de la réponse présents dans le contexte
- **Confiance globale** : Score combiné de la qualité de la réponse

## Prérequis

- Python 3.12 ou supérieur
- Clé API Google pour le modèle Gemini

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
streamlit run app.py
```

Le script vérifie automatiquement la présence des données et les télécharge si nécessaire :
- **PokeAPI** : Données Pokémon de la génération 1
- **Poképédia** : Scraping automatique des pages contenant "est un pokemon"
- **Index** : Construction automatique des index par type, statut, habitat, couleur

2. **Ouvrir l'interface** : `http://localhost:8501`

## Structure du projet

```
rag-lo17/
├── app.py                      # Application Streamlit
├── main.py                     # Script principal de lancement
├── src/
│   ├── rag_core.py            # Système RAG principal
│   ├── evaluation.py          # Métriques d'évaluation simplifiées
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

## Évaluation simplifiée

Le système utilise des métriques simples et efficaces pour évaluer la qualité des réponses :

### Indicateurs de confiance
- **Probabilité d'hallucination** : Risque que la réponse contienne des informations incorrectes
- **Recouvrement du contexte** : Pourcentage de mots de la réponse présents dans le contexte source
- **Confiance globale** : Score combiné de la qualité de la réponse

### Méthode de calcul
- **Similarité de séquences** : Comparaison structurelle entre réponse et contexte
- **Chevauchement de mots-clés** : Analyse des mots communs
- **Combinaison équilibrée** : Moyenne des deux scores pour un résultat réaliste

### Seuils d'alerte
- **Probabilité d'hallucination > 30%** : Avertissement automatique
- **Recouvrement faible** : Indicateur de risque d'hallucination
- **Confiance globale** : Barre de progression visuelle

## Lancement de l'évaluation

```bash
# Évaluation complète avec métriques détaillées
python src/evaluate_rag.py

# Évaluation en mode engagé
python src/evaluate_rag.py --engaged

# Évaluation avec questions personnalisées
python src/evaluate_rag.py --questions path/to/questions.txt
```

### Résultats d'évaluation
- **Rapport détaillé** : `evaluation_results/evaluation_report.txt`
- **Données brutes** : `evaluation_results/evaluation_results.csv`
- **Visualisations** : `evaluation_results/evaluation_metrics.png`

## Détection d'hallucinations

Le système détecte automatiquement les hallucinations potentielles :
- **Seuil de probabilité** : > 30% déclenche un avertissement
- **Indicateurs visuels** : Métriques de confiance en temps réel
- **Analyse du contexte** : Vérification de la cohérence avec les sources

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

- ✅ **Évaluation simplifiée** : Métriques de confiance claires et efficaces
- ✅ **Détection d'hallucinations** : Indicateurs en temps réel
- ✅ **Interface optimisée** : Focus sur l'expérience utilisateur
- ✅ **Intégration Poképédia** : Contenu enrichi en français
- ✅ **Métadonnées enrichies** : Index par type, statut, habitat, couleur
- ✅ **Prompts optimisés** : Instructions précises pour l'utilisation du contexte
- ✅ **Correction protobuf** : Compatibilité avec les nouvelles versions
- ✅ **Gestion d'erreurs** : Robustesse améliorée

## Contribution

1. Fork le repository
2. Créer une branche feature
3. Commiter les changements
4. Pousser vers la branche
5. Créer une pull request

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
