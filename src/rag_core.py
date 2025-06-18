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

###############################################################################
# HybridIndex (inchangé)
###############################################################################

class HybridIndex:
    def __init__(self, indexes_dir: str = "data/indexes"):
        self.indexes_dir = Path(indexes_dir)
        self.indexes: Dict[str, Dict[str, List[str]]] = {}
        self._load_indexes()

    def _load_indexes(self):
        """Charge les index de type / statut / habitat / couleur."""
        index_files = {
            "type": "type_index.json",
            "status": "status_index.json",
            "evolution": "evolution_index.json",
            "habitat": "habitat_index.json",
            "color": "color_index.json",
        }
        for index_name, filename in index_files.items():
            path = self.indexes_dir / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.indexes[index_name] = json.load(f)

    # Méthodes de recherche exactes -------------------------------------------------
    def search_by_type(self, type_name: str) -> List[str]:
        return self.indexes.get("type", {}).get(type_name, [])

    def search_by_status(self, status: str) -> List[str]:
        return self.indexes.get("status", {}).get(status, [])

    def search_by_habitat(self, habitat: str) -> List[str]:
        return self.indexes.get("habitat", {}).get(habitat, [])

    def search_by_color(self, color: str) -> List[str]:
        return self.indexes.get("color", {}).get(color, [])

###############################################################################
# RAGSystem : version "précise & concise"
###############################################################################

class RAGSystem:
    """Retrieval‑Augmented Generation (Pokémon).

    Cette version est optimisée pour fournir des réponses concises : le prompt
    impose une limite de longueur et la configuration LLM bride le nombre de
    tokens générés. Par défaut, on cherche 2 documents (k=2) pour réduire la
    surcharge contextuelle.
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

        # Embeddings & LLM --------------------------------------------------------
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        
        # Ajuster les tokens selon le mode
        if engaged_mode:
            max_tokens = max(max_tokens, 512)  # Plus de tokens pour le mode engagé
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,  # Limite la longueur de sortie
        )

        # Stores / Retriever ------------------------------------------------------
        self.vectorstore = None
        self.retriever = None
        self.hybrid_index = HybridIndex()

        # Prompt : ton neutre et concis ------------------------------------------
        self._update_prompt_template()

    # ---------------------------------------------------------------------------
    # Ingestion des documents
    # ---------------------------------------------------------------------------
    def embed_documents(self, documents: List[Document]) -> None:
        """Vectorise et indexe la liste de Documents dans Chroma."""
        from langchain.embeddings.base import Embeddings
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            # Ajuster k selon le mode
            k_value = 4 if self.engaged_mode else 2  # Plus de contexte pour le mode engagé
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_value})
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(f"Erreur d'intégration des documents : {exc}") from exc

    # ---------------------------------------------------------------------------
    # Nettoyage des ressources
    # ---------------------------------------------------------------------------
    def cleanup(self):
        """Supprime le dossier temporaire Chroma."""
        import shutil
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def __del__(self):
        self.cleanup()

    # ---------------------------------------------------------------------------
    # Helpers internes
    # ---------------------------------------------------------------------------
    def _update_prompt_template(self):
        """Met à jour le prompt template selon le mode engagé."""
        if self.engaged_mode:
            self.prompt_template = PromptTemplate.from_template(
                """You are a Pokémon encyclopedia assistant. Provide accurate, concise, and well-structured information about Pokémon.
                Rely only on the context below to answer the question. If the answer is not in the context, respond with "I don't know".

                Response guidelines:
                1. Be concise and strictly informative.
                2. Prioritize accuracy; avoid unnecessary anecdotes or trivia.
                3. Structure the answer in clear, logical paragraphs.
                4. Use a neutral, professional tone.
                5. If the context contains Poképédia content, incorporate it explicitly.
                6. For broad questions, give a concise, ordered overview.
                7. For specific questions, address only the relevant details.

                Question: {question}
                Context: {context}
                Answer:"""
                        )
        else:
            self.prompt_template = PromptTemplate.from_template(
                """You are a Pokémon encyclopedia assistant. 
                Provide accurate and concise answers strictly relevant to the question. 
                Use only the context below. 
                If the answer is not in the context, reply with \"I don't know\".
                Response guidelines:
                1. Keep answers short – ideally 2‑4 sentences (≈120 words max).
                2. State only essential facts; omit tangential trivia.
                3. Use a neutral, professional tone.
                4. Integrate Poképédia content when present.
                5. For broad questions, give a bullet‑style overview.
                6. For specific questions, answer directly without additional commentary.
                Question: {question}
                Context: {context}
                Answer:"""
            )
        
        # Mettre à jour la configuration du retriever si il existe
        if self.retriever and hasattr(self.retriever, 'search_kwargs'):
            k_value = 4 if self.engaged_mode else 2
            self.retriever.search_kwargs["k"] = k_value

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

    # ---------------------------------------------------------------------------
    # API publique
    # ---------------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """Interroge le système ; renvoie answer + context + metadata."""
        if not self.retriever:
            raise ValueError("Aucun document n'a été intégré (retriever non initialisé).")

        question_lower = question.lower()

        # -----------------------------------
        # 1. Recherche exacte via HybridIndex
        # -----------------------------------
        # (inchangé, mais gardé pour complétude)
        if any(kw in question_lower for kw in ["type", "est de type", "sont de type", "liste", "quels sont"]):
            for type_name in self.hybrid_index.indexes.get("type", {}):
                if type_name in question_lower:
                    pk_names = self.hybrid_index.search_by_type(type_name)
                    if pk_names:
                        return {
                            "answer": f"Les Pokémon de type {type_name} : {', '.join(pk_names)}",
                            "context": [],
                            "metadata": [],
                            "search_type": "exact",
                        }

        if any(kw in question_lower for kw in ["légendaire", "legendary", "légendaires", "legendaries"]):
            pk_names = self.hybrid_index.search_by_status("legendary")
            if pk_names:
                return {
                    "answer": f"Les Pokémon légendaires : {', '.join(pk_names)}",
                    "context": [],
                    "metadata": [],
                    "search_type": "exact",
                }

        if any(kw in question_lower for kw in ["mythique", "mythical", "mythiques", "mythicals"]):
            pk_names = self.hybrid_index.search_by_status("mythical")
            if pk_names:
                return {
                    "answer": f"Les Pokémon mythiques : {', '.join(pk_names)}",
                    "context": [],
                    "metadata": [],
                    "search_type": "exact",
                }

        # -------------------------------------
        # 2. Recherche sémantique (LLM + RAG)
        # -------------------------------------
        try:
            docs = self.retriever.invoke(question)
            answer_chain = self._build_chain()
            answer = answer_chain.invoke(question)
            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs],
                "metadata": [doc.metadata for doc in docs],
                "search_type": "semantic",
            }
        except Exception as exc:
            # En cas d'erreur, on réinitialise Chroma pour éviter les corruptions
            self.vectorstore = None
            self.retriever = None
            raise RuntimeError(f"Erreur durant la recherche : {exc}") from exc
