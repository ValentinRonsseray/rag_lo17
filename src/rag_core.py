import os
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# Clé API Google (indispensable pour Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY non trouvée")

from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def load_pokepedia_documents() -> List[Document]:
    """Charge et formate les documents Poképédia."""
    pokepedia_dir = Path("data/pokepedia")
    documents = []
    
    if not pokepedia_dir.exists():
        print("Dossier Poképédia non trouvé, création d'un exemple...")
        return documents
    
    for json_file in pokepedia_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extraire le nom du Pokémon depuis le nom du fichier
            pokemon_name = json_file.stem
            
            # Formater le contenu
            content = data.get("content", "")
            if content:
                # Créer un document avec métadonnées
                doc = Document(
                    page_content=f"Informations Poképédia sur {pokemon_name}:\n\n{content}",
                    metadata={
                        "source": "pokepedia",
                        "pokemon_name": pokemon_name,
                        "url": data.get("url", ""),
                        "timestamp": data.get("timestamp", ""),
                        "content_type": "pokepedia_description"
                    }
                )
                documents.append(doc)
                print(f"Document Poképédia chargé: {pokemon_name}")
                
        except Exception as e:
            print(f"Erreur lors du chargement de {json_file}: {e}")
    
    print(f"Total documents Poképédia chargés: {len(documents)}")
    return documents

def load_index_data() -> Dict[str, Dict[str, List[str]]]:
    """Charge les données d'index depuis les fichiers JSON."""
    indexes_dir = Path("data/indexes")
    indexes = {}
    
    if not indexes_dir.exists():
        print("Dossier indexes non trouvé")
        return indexes
    
    index_files = {
        "type": "type_index.json",
        "status": "status_index.json",
        "habitat": "habitat_index.json",
        "color": "color_index.json",
    }
    
    for index_name, filename in index_files.items():
        path = indexes_dir / filename
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    indexes[index_name] = json.load(f)
                print(f"Index {index_name} chargé: {len(indexes[index_name])} catégories")
            except Exception as e:
                print(f"Erreur lors du chargement de l'index {index_name}: {e}")
    
    return indexes

# RAGSystem : version simplifiée sans recherche hybride

class RAGSystem:
    """Retrieval‑Augmented Generation (Pokémon).

    Cette version utilise uniquement la recherche vectorielle avec des métadonnées enrichies
    incluant les informations d'index pour une recherche plus précise.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "models/embedding-001",
        temperature: float = 0.0,
        max_tokens: int = 256,
        engaged_mode: bool = False,
    ) -> None:
        import tempfile
        # Dossier temporaire pour la BDD Chroma
        self.persist_directory = Path(tempfile.mkdtemp(prefix="chroma_db_"))
        
        # Mode engagé
        self.engaged_mode = engaged_mode

        # Embeddings & LLM 
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        
        # Ajuster les tokens selon le mode
        if engaged_mode:
            max_tokens = max(max_tokens, 512)  # Plus de tokens pour le mode engagé
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,  # Limite la longueur de sortie
        )

        # Stores / Retriever 
        self.vectorstore = None
        self.retriever = None

        # Prompt : ton neutre et concis 
        self._update_prompt_template()

    # Ingestion des documents
    def embed_documents(self, documents: List[Document], pokepedia_documents: List[Document] = None) -> None:
        """Vectorise et indexe la liste de Documents dans Chroma."""
        from langchain.embeddings.base import Embeddings
        
        # Charger les documents Poképédia si pas fournis
        if pokepedia_documents is None:
            pokepedia_documents = load_pokepedia_documents()
        
        # Charger les données d'index
        indexes = load_index_data()
        
        # Enrichir les métadonnées des documents avec les informations d'index
        enriched_documents = self._enrich_documents_with_indexes(documents, indexes)
        enriched_pokepedia = self._enrich_documents_with_indexes(pokepedia_documents, indexes)
        
        # Combiner tous les documents
        all_documents = enriched_documents + enriched_pokepedia
        
        print(f"Intégration de {len(documents)} documents PokeAPI + {len(pokepedia_documents)} documents Poképédia")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            # Ajuster k selon le mode
            k_value = 4 if self.engaged_mode else 2  # Plus de contexte pour le mode engagé
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_value})
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(f"Erreur d'intégration des documents : {exc}") from exc

    def _enrich_documents_with_indexes(self, documents: List[Document], indexes: Dict[str, Dict[str, List[str]]]) -> List[Document]:
        """Enrichit les documents avec les informations d'index."""
        enriched_docs = []
        
        for doc in documents:
            pokemon_name = doc.metadata.get("name", "").lower()
            if not pokemon_name:
                # Essayer de récupérer le nom depuis le contenu pour les documents Poképédia
                if doc.metadata.get("source") == "pokepedia":
                    pokemon_name = doc.metadata.get("pokemon_name", "").lower()
            
            if pokemon_name:
                # Ajouter les informations d'index aux métadonnées
                enriched_metadata = doc.metadata.copy()
                
                # Types - convertir en chaîne
                pokemon_types = []
                for type_name, pokemon_list in indexes.get("type", {}).items():
                    if pokemon_name in pokemon_list:
                        pokemon_types.append(type_name)
                if pokemon_types:
                    enriched_metadata["pokemon_types"] = ", ".join(pokemon_types)
                
                # Statut (légendaire, mythique, bébé)
                for status, pokemon_list in indexes.get("status", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata[f"is_{status}"] = True
                
                # Habitat
                for habitat_name, pokemon_list in indexes.get("habitat", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata["habitat"] = habitat_name
                
                # Couleur
                for color_name, pokemon_list in indexes.get("color", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata["color"] = color_name
                
                # Filtrer manuellement les métadonnées complexes
                filtered_metadata = {}
                for key, value in enriched_metadata.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered_metadata[key] = value
                    elif isinstance(value, list):
                        # Convertir les listes en chaînes
                        filtered_metadata[key] = ", ".join(str(item) for item in value)
                    elif isinstance(value, dict):
                        # Convertir les dictionnaires en chaînes JSON
                        import json
                        filtered_metadata[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        # Convertir les autres types en chaînes
                        filtered_metadata[key] = str(value)
                
                # Créer un nouveau document avec les métadonnées enrichies
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                enriched_docs.append(enriched_doc)
            else:
                # Garder le document original si pas de nom trouvé
                enriched_docs.append(doc)
        
        return enriched_docs

   
    def cleanup(self):
        """Supprime le dossier temporaire Chroma."""
        import shutil
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def __del__(self):
        self.cleanup()

    # Helpers 
    def _update_prompt_template(self):
        """Met à jour le prompt template selon le mode engagé."""
        if self.engaged_mode:
            self.prompt_template = PromptTemplate.from_template(
                """You are a Pokémon encyclopedia assistant. Your task is to provide accurate, comprehensive, and well-structured information about Pokémon based EXCLUSIVELY on the context provided below.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context. If the answer is not in the context, respond with "I don't have enough information to answer this question accurately."
2. SEARCH THOROUGHLY through the context for relevant information before answering.
3. USE ALL AVAILABLE CONTEXT - do not ignore any relevant details.
4. CITE SPECIFIC INFORMATION from the context when possible.
5. DISTINGUISH between different data sources in the context (PokeAPI vs Poképédia).

CONTEXT ANALYSIS GUIDELINES:
- For statistical questions (stats, types, abilities, evolution): Look for PokeAPI data first
- For descriptive questions (appearance, behavior, lore): Look for Poképédia data first
- For categorization questions (lists, types, habitats): Use metadata indexes when available
- For comparison questions: Extract specific values from the context and compare them
- For detailed descriptions: Combine information from multiple context sources

RESPONSE STRUCTURE:
1. Start with a direct answer to the question
2. Provide specific details from the context
3. Mention the source of information when relevant
4. Structure information logically (most important first)
5. Include numerical data when available in the context

CONTEXT SOURCES TO USE:
- PokeAPI data: Technical specifications, statistics, types, abilities, evolution chains
- Poképédia data: Descriptions, biology, behavior, habitat, mythology, cultural aspects
- Metadata indexes: pokemon_types, is_legendary, is_mythical, habitat, color information

Question: {question}
Context: {context}

Answer:"""
            )
        else:
            self.prompt_template = PromptTemplate.from_template(
                """You are a Pokémon encyclopedia assistant. Provide accurate and concise answers based EXCLUSIVELY on the context provided below.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context. If the answer is not in the context, respond with "I don't have enough information to answer this question accurately."
2. SEARCH THOROUGHLY through the context for relevant information.
3. USE ALL AVAILABLE CONTEXT - do not ignore relevant details.
4. BE SPECIFIC - cite exact values, names, and details from the context.

CONTEXT SEARCH STRATEGY:
- For statistics questions: Look for PokeAPI data with specific numbers
- For description questions: Look for Poképédia content with detailed explanations
- For list questions: Use metadata indexes and context information
- For comparison questions: Extract and compare specific values from context
- For general questions: Combine the most relevant information from all sources

RESPONSE GUIDELINES:
1. Keep answers concise but informative (3-5 sentences)
2. Start with the most important information
3. Include specific details from the context
5. Use exact values and names from the context

CONTEXT SOURCES:
- PokeAPI: Statistics, types, abilities, technical data
- Poképédia: Descriptions, behavior, habitat, lore
- Metadata: Type information, legendary status, habitat, color

Question: {question}
Context: {context}

Answer:"""
            )
        
        # Mettre à jour la configuration du retriever si il existe
        if self.retriever and hasattr(self.retriever, 'search_kwargs'):
            k_value = 4 if self.engaged_mode else 2
            self.retriever.search_kwargs["k"] = k_value

    def update_temperature(self, temperature: float):
        """Met à jour la température du modèle LLM."""
        self.llm.temperature = temperature
        print(f"🌡️ Température mise à jour: {temperature}")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Chaîne RAG (Retriever → Prompt → LLM)."""
        return (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    # API publique
    def query(self, question: str) -> Dict[str, Any]:
        """Interroge le système ; renvoie answer + context + metadata."""
        if not self.retriever:
            raise ValueError("Aucun document n'a été intégré (retriever non initialisé).")

        # Debug console - Affichage des informations de requête
        print("=" * 60)
        print("DEBUG RAG - NOUVELLE REQUÊTE")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Température: {self.llm.temperature}")
        print(f"Max tokens: {self.llm.max_output_tokens}")
        print(f"Modèle: {self.llm.model}")
        print(f"Mode engagé: {self.engaged_mode}")
        print(f"K documents: {self.retriever.search_kwargs.get('k', 'N/A')}")
        print("-" * 60)

        # Recherche sémantique (LLM + RAG)
        print("Recherche sémantique (RAG) en cours...")
        try:
            docs = self.retriever.invoke(question)
            print(f"Documents récupérés: {len(docs)}")
            
            answer_chain = self._build_chain()
            answer = answer_chain.invoke(question)
            
            print(f"Réponse générée: {len(answer)} caractères")
            print("=" * 60)
            
            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs],
                "metadata": [doc.metadata for doc in docs],
                "search_type": "semantic",
            }
        except Exception as exc:
            print(f"ERREUR: {exc}")
            print("=" * 60)
            # En cas d'erreur, on réinitialise Chroma pour éviter les corruptions
            self.vectorstore = None
            self.retriever = None
            raise RuntimeError(f"Erreur durant la recherche : {exc}") from exc
