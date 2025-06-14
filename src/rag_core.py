import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# charge les variables d'environnement
load_dotenv()

# récupère la clé api
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY non trouvée")

from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class HybridIndex:
    def __init__(self, indexes_dir: str = "data/indexes"):
        self.indexes_dir = Path(indexes_dir)
        self.indexes = {}
        self.load_indexes()
    
    def load_indexes(self):
        """charge les index"""
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
        """recherche par type"""
        return self.indexes.get("type", {}).get(type_name, [])
    
    def search_by_status(self, status: str) -> List[str]:
        """recherche par statut"""
        return self.indexes.get("status", {}).get(status, [])
    
    def search_by_habitat(self, habitat: str) -> List[str]:
        """recherche par habitat"""
        return self.indexes.get("habitat", {}).get(habitat, [])
    
    def search_by_color(self, color: str) -> List[str]:
        """recherche par couleur"""
        return self.indexes.get("color", {}).get(color, [])

class RAGSystem:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "models/embedding-001",
        temperature: float = 0.0,
    ):
        # dossier temporaire pour la bdd
        import tempfile
        self.persist_directory = Path(tempfile.mkdtemp(prefix="chroma_db_"))
        
        # init des modèles
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        # init du stockage et de l'index
        self.vectorstore = None
        self.retriever = None
        self.hybrid_index = HybridIndex()
        
        # template du prompt
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
        """intègre les documents dans chroma"""
        try:
            # crée une nouvelle bdd
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # sauvegarde la bdd
            self.vectorstore.persist()
        except Exception as e:
            print(f"erreur d'intégration : {e}")
            # nettoie en cas d'erreur
            self.cleanup()
            raise
    
    def cleanup(self):
        """nettoie les ressources"""
        try:
            if self.persist_directory.exists():
                import shutil
                shutil.rmtree(self.persist_directory, ignore_errors=True)
        except Exception as e:
            print(f"erreur de nettoyage : {e}")
    
    def __del__(self):
        """destructeur"""
        self.cleanup()
    
    def format_docs(self, docs: List[Document]) -> str:
        """formate les docs pour le prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self):
        """crée la chaîne rag"""
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """interroge le système rag"""
        if not self.retriever:
            raise ValueError("pas de documents intégrés")
        
        # analyse de la question
        question_lower = question.lower()
        
        # recherche par type
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
        
        # recherche par statut
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
        
        # recherche vectorielle par défaut
        try:
            docs = self.retriever.invoke(question)
            chain = self.create_chain()
            answer = chain.invoke(question)
            
            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs],
                "metadata": [doc.metadata for doc in docs],
                "search_type": "semantic"
            }
        except Exception as e:
            print(f"erreur de recherche : {e}")
            # réinitialise la bdd en cas d'erreur
            self.vectorstore = None
            self.retriever = None
            raise ValueError("erreur de la bdd vectorielle") 