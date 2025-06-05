"""
Core RAG components for the question answering system.
"""

import os
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
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use five sentences maximum and keep the answer concise.

            Question: {question} 
            Context: {context} 
            Answer:"""
        )
        
    def load_documents(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs.
        
        Args:
            urls: List of URLs to load documents from
            
        Returns:
            List of loaded documents
        """
        loader = WebBaseLoader(urls)
        return loader.load()
    
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
            
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Generate answer
        chain = self.create_chain()
        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs]
        } 