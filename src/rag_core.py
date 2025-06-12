"""
Core RAG components for the question answering system.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file")

from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class HybridIndex:
    """
    Index hybride combinant recherche vectorielle et index inverses.
    
    Avantages de l'index hybride :
    1. Recherche sémantique : La recherche vectorielle permet de trouver des documents
       pertinents même si les termes exacts ne sont pas présents.
    2. Recherche exacte : Les index inverses permettent de trouver rapidement des
       Pokémon par leurs caractéristiques spécifiques (type, statut, etc.).
    3. Performance : Les index inverses sont très rapides pour les requêtes exactes.
    4. Flexibilité : Permet de combiner les deux approches selon le type de requête.
    """
    
    def __init__(self, indexes_dir: str = "data/indexes"):
        """Initialise l'index hybride.
        
        Args:
            indexes_dir: Répertoire contenant les index inverses
        """
        self.indexes_dir = Path(indexes_dir)
        self.indexes = {}
        self.load_indexes()
    
    def load_indexes(self):
        """Charge tous les index inverses."""
        index_files = {
            "type": "type_index.json",
            "status": "status_index.json",
            "evolution": "evolution_index.json",
            "habitat": "habitat_index.json",
            "color": "color_index.json"
        }
        
        for index_name, filename in index_files.items():
            file_path = self.indexes_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.indexes[index_name] = json.load(f)
    
    def search_by_type(self, type_name: str) -> List[str]:
        """Recherche les Pokémon par type."""
        return self.indexes.get("type", {}).get(type_name, [])
    
    def search_by_status(self, status: str) -> List[str]:
        """Recherche les Pokémon par statut (légendaire, mythique, bébé)."""
        return self.indexes.get("status", {}).get(status, [])
    
    def search_by_habitat(self, habitat: str) -> List[str]:
        """Recherche les Pokémon par habitat."""
        return self.indexes.get("habitat", {}).get(habitat, [])
    
    def search_by_color(self, color: str) -> List[str]:
        """Recherche les Pokémon par couleur."""
        return self.indexes.get("color", {}).get(color, [])

class RAGSystem:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "models/embedding-001",
        temperature: float = 0.0,
    ):
        """Initialize the RAG system.
        
        Args:
            persist_directory: Directory to persist the vector store
            model_name: Name of the Gemini model to use
            embedding_model: Name of the embedding model to use
            temperature: Temperature for generation
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        # Initialize vector store and hybrid index
        self.vectorstore = None
        self.retriever = None
        self.hybrid_index = HybridIndex()
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are an expert Pokémon encyclopedia assistant. Your role is to provide detailed and engaging information about Pokémon.
            Use the following context to answer the question. If you don't know the answer, just say that you don't know.
            
            Guidelines for your response:
            1. Be informative and engaging
            2. Include interesting details and trivia when available
            3. Structure your response in clear paragraphs
            4. Use natural, conversational language
            5. If the context contains Poképédia information, make sure to include it
            6. For general questions, provide a comprehensive overview
            7. For specific questions, focus on the relevant details
            
            Question: {question} 
            Context: {context} 
            Answer:"""
        )
    
    def embed_documents(self, documents: List[Document]) -> None:
        """Embed documents and store them in Chroma.
        
        Args:
            documents: List of documents to embed
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format documents for the prompt.
        
        Args:
            docs: List of documents to format
            
        Returns:
            Formatted string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self):
        """Create the RAG chain."""
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary containing the answer and retrieved context
        """
        if not self.retriever:
            raise ValueError("No documents have been embedded yet. Call embed_documents first.")
        
        # Analyse de la question pour déterminer le type de recherche
        question_lower = question.lower()
        
        # Recherche par type
        if any(keyword in question_lower for keyword in ["type", "est de type", "sont de type", "liste", "quels sont"]):
            for type_name in self.hybrid_index.indexes.get("type", {}).keys():
                if type_name in question_lower:
                    pokemon_names = self.hybrid_index.search_by_type(type_name)
                    if pokemon_names:
                        return {
                            "answer": f"Les Pokémon de type {type_name} sont : {', '.join(pokemon_names)}",
                            "context": [],
                            "metadata": [],
                            "search_type": "exact"
                        }
        
        # Recherche par statut
        if any(keyword in question_lower for keyword in ["légendaire", "legendary", "légendaires", "legendaries"]):
            pokemon_names = self.hybrid_index.search_by_status("legendary")
            if pokemon_names:
                return {
                    "answer": f"Les Pokémon légendaires sont : {', '.join(pokemon_names)}",
                    "context": [],
                    "metadata": [],
                    "search_type": "exact"
                }
        
        if any(keyword in question_lower for keyword in ["mythique", "mythical", "mythiques", "mythicals"]):
            pokemon_names = self.hybrid_index.search_by_status("mythical")
            if pokemon_names:
                return {
                    "answer": f"Les Pokémon mythiques sont : {', '.join(pokemon_names)}",
                    "context": [],
                    "metadata": [],
                    "search_type": "exact"
                }
        
        # Si aucune recherche exacte n'est possible, utiliser la recherche vectorielle
        docs = self.retriever.invoke(question)  # Utiliser invoke au lieu de get_relevant_documents
        chain = self.create_chain()
        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs],
            "search_type": "semantic"
        } 