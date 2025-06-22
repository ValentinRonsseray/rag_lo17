# rag pokémon question answering system

un système de questions-réponses basé sur rag (retrieval-augmented generation) spécialisé dans l'univers pokémon, utilisant langchain, le modèle gemini de google et chromadb.

## fonctionnalités

- **sources de données multiples** : pokeapi (statistiques, types, capacités) + poképédia (descriptions, biologie, comportement)
- **recherche vectorielle avancée** avec métadonnées enrichies (types, statuts, habitats, couleurs)
- **mode de réponse configurable** : normal (concis) ou engagé (détaillé)
- **évaluation complète** avec métriques inspirées de ragas
- **détection d'hallucinations** et journalisation automatique
- **interface web streamlit** intuitive
- **scraping automatique** des données poképédia
- **index hybrides** pour une recherche optimisée

## architecture

### sources de données
- **pokeapi** : données techniques officielles (statistiques, types, capacités, évolutions)
- **poképédia** : contenu enrichi en français (descriptions, biologie, comportement, habitat, mythologie)
- **index métadonnées** : types, statuts légendaires, habitats, couleurs

### système rag
- **embeddings** : google generative ai (models/embedding-001)
- **llm** : gemini 2.0 flash
- **vector store** : chromadb avec métadonnées enrichies
- **recherche** : vectorielle sémantique avec filtrage par métadonnées

## prérequis

- python 3.12 ou supérieur
- clé api google pour le modèle gemini

## installation

1. **cloner le repository** :
```bash
git clone <repository-url>
cd rag-lo17
```

2. **installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **configurer la clé api google** :
```bash
# créer un fichier .env
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

## utilisation

### lancement local

1. **lancer l'application** :
```bash
python main.py --run
```

le script vérifie automatiquement la présence des données et les télécharge si nécessaire :
- **pokeapi** : données pokémon de la génération 1
- **poképédia** : scraping automatique des pages contenant "est un pokemon"
- **index** : construction automatique des index par type, statut, habitat, couleur

2. **ouvrir l'interface** : `http://localhost:8501`

## structure du projet

```
rag-lo17/
├── app.py                      # application streamlit
├── main.py                     # script principal de lancement
├── src/
│   ├── rag_core.py            # système rag principal
│   ├── evaluation.py          # métriques d'évaluation
│   ├── evaluate_rag.py        # script d'évaluation complet
│   ├── scrap_pokeapi.py       # téléchargement pokeapi
│   ├── scrap_pokepedia.py     # scraping poképédia
│   ├── build_pokemon_index.py # construction des index
│   ├── format_pokeapi_data.py # formatage des données
│   └── pokepedia_data.py      # gestion des données poképédia
├── data/
│   ├── pokeapi/               # données pokeapi
│   ├── pokepedia/             # données poképédia
│   └── indexes/               # index de recherche
├── evaluation_results/        # résultats d'évaluation
├── requirements.txt           # dépendances python
└── readme.md                  # documentation
```

## fonctionnalités avancées

### mode de réponse
- **mode normal** : réponses concises (3-5 phrases)
- **mode engagé** : réponses détaillées et structurées

### types de questions supportées
- **statistiques** : "quelles sont les stats de pikachu ?"
- **descriptions** : "décris le comportement de charizard"
- **catégorisation** : "quels sont les pokémon de type feu ?"
- **comparaisons** : "qui est plus rapide entre mew et machopeur ?"
- **lore et mythologie** : "raconte l'histoire de mewtwo"

### métadonnées enrichies
- **types** : pokemon_types (ex: "fire, flying")
- **statuts** : is_legendary, is_mythical, is_baby
- **habitat** : habitat (ex: "mountain", "forest")
- **couleur** : color (ex: "red", "blue")

## évaluation

le système inclut une évaluation complète avec métriques inspirées de ragas :

### métriques calculées
- **f1 score** : précision et rappel de la réponse
- **similarité** : similarité sémantique avec la référence
- **précision contexte** : pertinence du contexte récupéré
- **rappel contexte** : complétude du contexte par rapport à la référence
- **fidélité** : proportion de la réponse basée sur le contexte

### lancement de l'évaluation
```bash
# utiliser le jeu de questions par défaut
python src/evaluate_rag.py

# utiliser un jeu de questions personnalisé
python src/evaluate_rag.py path/to/questions.json
```

### résultats
- **rapport détaillé** : `evaluation_results/evaluation_report.txt`
- **données brutes** : `evaluation_results/evaluation_results.csv`
- **visualisations** : `evaluation_results/evaluation_metrics.png`

## détection d'hallucinations

le système détecte automatiquement les hallucinations potentielles :
- **seuil** : fidélité < 0.7
- **journalisation** : `hallucinations.csv`
- **indicateurs visuels** : barres de confiance dans l'interface

## configuration

### variables d'environnement
```bash
GOOGLE_API_KEY=your-api-key-here
```

### paramètres du modèle
- **température** : 0.0 (déterministe) à 1.0 (créatif)
- **max tokens** : 256 (normal) à 512 (engagé)
- **documents récupérés** : 2 (normal) à 4 (engagé)

## améliorations récentes

- ✅ **intégration poképédia** : contenu enrichi en français
- ✅ **métadonnées enrichies** : index par type, statut, habitat, couleur
- ✅ **prompts optimisés** : instructions précises pour l'utilisation du contexte
- ✅ **évaluation complète** : rapport détaillé avec statistiques
- ✅ **interface améliorée** : mode engagé par défaut, indicateurs de confiance
- ✅ **correction protobuf** : compatibilité avec les nouvelles versions

## contribution

1. fork le repository
2. créer une branche feature
3. commiter les changements
4. pousser vers la branche
5. créer une pull request

## licence

ce projet est sous licence mit - voir le fichier license pour plus de détails.
