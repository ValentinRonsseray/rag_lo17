import os
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# fix pour le problème de protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# fix pour le problème de distutils en python 3.12+
try:
    import distutils
except ImportError:
    import setuptools._distutils as distutils

# charge les variables d'environnement
load_dotenv()

# clé api google (indispensable pour gemini)
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
    """charge et formate les documents poképédia."""
    pokepedia_dir = Path("data/pokepedia")
    documents = []

    if not pokepedia_dir.exists():
        print("dossier poképédia non trouvé, création d'un exemple...")
        return documents

    for json_file in pokepedia_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # extraire le nom du pokémon depuis le nom du fichier
            pokemon_name = json_file.stem

            # formater le contenu
            content = data.get("content", "")
            if content:
                # créer un document avec métadonnées
                doc = Document(
                    page_content=f"informations poképédia sur {pokemon_name}:\n\n{content}",
                    metadata={
                        "source": "pokepedia",
                        "pokemon_name": pokemon_name,
                        "url": data.get("url", ""),
                        "timestamp": data.get("timestamp", ""),
                        "content_type": "pokepedia_description",
                    },
                )
                documents.append(doc)
                print(f"document poképédia chargé: {pokemon_name}")

        except Exception as e:
            print(f"erreur lors du chargement de {json_file}: {e}")

    print(f"total documents poképédia chargés: {len(documents)}")
    return documents


def load_index_data() -> Dict[str, Dict[str, List[str]]]:
    """charge les données d'index depuis les fichiers json."""
    indexes_dir = Path("data/indexes")
    indexes = {}

    if not indexes_dir.exists():
        print("dossier indexes non trouvé")
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
                print(
                    f"index {index_name} chargé: {len(indexes[index_name])} catégories"
                )
            except Exception as e:
                print(f"erreur lors du chargement de l'index {index_name}: {e}")

    return indexes


class RAGSystem:
    """retrieval‑augmented generation (pokémon).

    cette version utilise uniquement la recherche vectorielle avec des métadonnées enrichies
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

        # dossier temporaire pour la bdd chroma
        self.persist_directory = Path(tempfile.mkdtemp(prefix="chroma_db_"))

        # mode engagé
        self.engaged_mode = engaged_mode

        # embeddings & llm
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        # ajuster les tokens selon le mode
        if engaged_mode:
            max_tokens = max(max_tokens, 512)  # plus de tokens pour le mode engagé

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,  # limite la longueur de sortie
        )

        # stores / retriever
        self.vectorstore = None
        self.retriever = None

        # prompt : ton neutre et concis
        self._update_prompt_template()

    def embed_documents(
        self, documents: List[Document], pokepedia_documents: List[Document] = None
    ) -> None:
        """vectorise et indexe la liste de documents dans chroma."""
        from langchain.embeddings.base import Embeddings

        # charger les documents poképédia si pas fournis
        if pokepedia_documents is None:
            pokepedia_documents = load_pokepedia_documents()

        # charger les données d'index
        indexes = load_index_data()

        # enrichir les métadonnées des documents avec les informations d'index
        enriched_documents = self._enrich_documents_with_indexes(documents, indexes)
        enriched_pokepedia = self._enrich_documents_with_indexes(
            pokepedia_documents, indexes
        )

        # combiner tous les documents
        all_documents = enriched_documents + enriched_pokepedia

        print(
            f"intégration de {len(documents)} documents pokeapi + {len(pokepedia_documents)} documents poképédia"
        )

        try:
            self.vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            # ajuster k selon le mode
            k_value = (
                4 if self.engaged_mode else 2
            )  # plus de contexte pour le mode engagé
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_value})
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(f"erreur d'intégration des documents : {exc}") from exc

    def _enrich_documents_with_indexes(
        self, documents: List[Document], indexes: Dict[str, Dict[str, List[str]]]
    ) -> List[Document]:
        """enrichit les documents avec les informations d'index."""
        enriched_docs = []

        for doc in documents:
            pokemon_name = doc.metadata.get("name", "").lower()
            if not pokemon_name:
                # essayer de récupérer le nom depuis le contenu pour les documents poképédia
                if doc.metadata.get("source") == "pokepedia":
                    pokemon_name = doc.metadata.get("pokemon_name", "").lower()

            if pokemon_name:
                # ajouter les informations d'index aux métadonnées
                enriched_metadata = doc.metadata.copy()

                # types - convertir en chaîne
                pokemon_types = []
                for type_name, pokemon_list in indexes.get("type", {}).items():
                    if pokemon_name in pokemon_list:
                        pokemon_types.append(type_name)
                if pokemon_types:
                    enriched_metadata["pokemon_types"] = ", ".join(pokemon_types)

                # statut (légendaire, mythique, bébé)
                for status, pokemon_list in indexes.get("status", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata[f"is_{status}"] = True

                # habitat
                for habitat_name, pokemon_list in indexes.get("habitat", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata["habitat"] = habitat_name

                # couleur
                for color_name, pokemon_list in indexes.get("color", {}).items():
                    if pokemon_name in pokemon_list:
                        enriched_metadata["color"] = color_name

                # filtrer manuellement les métadonnées complexes
                filtered_metadata = {}
                for key, value in enriched_metadata.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered_metadata[key] = value
                    elif isinstance(value, list):
                        # convertir les listes en chaînes
                        filtered_metadata[key] = ", ".join(str(item) for item in value)
                    elif isinstance(value, dict):
                        # convertir les dictionnaires en chaînes json
                        import json

                        filtered_metadata[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        # convertir les autres types en chaînes
                        filtered_metadata[key] = str(value)

                # créer un nouveau document avec les métadonnées enrichies
                enriched_doc = Document(
                    page_content=doc.page_content, metadata=filtered_metadata
                )
                enriched_docs.append(enriched_doc)
            else:
                # garder le document original si pas de nom trouvé
                enriched_docs.append(doc)

        return enriched_docs

    def cleanup(self):
        """supprime le dossier temporaire chroma."""
        import shutil

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def __del__(self):
        self.cleanup()

    def _update_prompt_template(self):
        """met à jour le prompt template selon le mode engagé."""
        if self.engaged_mode:
            self.prompt_template = PromptTemplate.from_template(
                """you are a pokémon encyclopedia assistant. your task is to provide accurate, comprehensive, and well-structured information about pokémon based exclusively on the context provided below.

critical instructions:
1. only use information from the provided context. if the answer is not in the context, respond with "i don't have enough information to answer this question accurately."
2. search thoroughly through the context for relevant information before answering.
3. use all available context - do not ignore any relevant details.
4. cite specific information from the context when possible.
5. distinguish between different data sources in the context (pokeapi vs poképédia).

context analysis guidelines:
- for statistical questions (stats, types, abilities, evolution): look for pokeapi data first
- for descriptive questions (appearance, behavior, lore): look for poképédia data first
- for categorization questions (lists, types, habitats): use metadata indexes when available
- for comparison questions: extract specific values from the context and compare them
- for detailed descriptions: combine information from multiple context sources

response structure:
1. start with a direct answer to the question
2. provide specific details from the context
3. mention the source of information when relevant
4. structure information logically (most important first)
5. include numerical data when available in the context

context sources to use:
- pokeapi data: technical specifications, statistics, types, abilities, evolution chains
- poképédia data: descriptions, biology, behavior, habitat, mythology, cultural aspects
- metadata indexes: pokemon_types, is_legendary, is_mythical, habitat, color information

question: {question}
context: {context}

answer:"""
            )
        else:
            self.prompt_template = PromptTemplate.from_template(
                """you are a pokémon encyclopedia assistant. provide accurate and concise answers based exclusively on the context provided below.

critical instructions:
1. only use information from the provided context. if the answer is not in the context, respond with "i don't have enough information to answer this question accurately."
2. search thoroughly through the context for relevant information.
3. use all available context - do not ignore relevant details.
4. be specific - cite exact values, names, and details from the context.

context search strategy:
- for statistics questions: look for pokeapi data with specific numbers
- for description questions: look for poképédia content with detailed explanations
- for list questions: use metadata indexes and context information
- for comparison questions: extract and compare specific values from context
- for general questions: combine the most relevant information from all sources

response guidelines:
1. keep answers concise but informative (3-5 sentences)
2. start with the most important information
3. include specific details from the context
5. use exact values and names from the context

context sources:
- pokeapi: statistics, types, abilities, technical data
- poképédia: descriptions, behavior, habitat, lore
- metadata: type information, legendary status, habitat, color

question: {question}
context: {context}

answer:"""
            )

        # mettre à jour la configuration du retriever si il existe
        if self.retriever and hasattr(self.retriever, "search_kwargs"):
            k_value = 4 if self.engaged_mode else 2
            self.retriever.search_kwargs["k"] = k_value

    def update_temperature(self, temperature: float):
        """met à jour la température du modèle llm."""
        self.llm.temperature = temperature
        print(f"🌡️ température mise à jour: {temperature}")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """chaîne rag (retriever → prompt → llm)."""
        return (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Any]:
        """interroge le système ; renvoie answer + context + metadata."""
        if not self.retriever:
            raise ValueError(
                "aucun document n'a été intégré (retriever non initialisé)."
            )

        # debug console - affichage des informations de requête
        print("=" * 60)
        print("debug rag - nouvelle requête")
        print("=" * 60)
        print(f"question: {question}")
        print(f"température: {self.llm.temperature}")
        print(f"max tokens: {self.llm.max_output_tokens}")
        print(f"modèle: {self.llm.model}")
        print(f"mode engagé: {self.engaged_mode}")
        print(f"k documents: {self.retriever.search_kwargs.get('k', 'n/a')}")
        print("-" * 60)

        # recherche sémantique (llm + rag)
        print("recherche sémantique (rag) en cours...")
        try:
            docs = self.retriever.invoke(question)
            print(f"documents récupérés: {len(docs)}")

            answer_chain = self._build_chain()
            answer = answer_chain.invoke(question)

            print(f"réponse générée: {len(answer)} caractères")
            print("=" * 60)

            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs],
                "metadata": [doc.metadata for doc in docs],
                "search_type": "semantic",
            }
        except Exception as exc:
            print(f"erreur: {exc}")
            print("=" * 60)
            # en cas d'erreur, on réinitialise chroma pour éviter les corruptions
            self.vectorstore = None
            self.retriever = None
            raise RuntimeError(f"erreur durant la recherche : {exc}") from exc
