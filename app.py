"""
Streamlit application for the RAG system.
"""

import os
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
from langchain.docstore.document import Document

from src.rag_core import RAGSystem
from src.evaluation import RAGEvaluator

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()

# Load and embed Pokemon data
if "pokemon_data_embedded" not in st.session_state:
    # Read Pokemon data
    pokemon_df = pd.read_csv("data/pokemon_basic_stats.csv")
    
    # Convert each Pokemon row into a document
    documents = []
    for _, row in pokemon_df.iterrows():
        # Create a text description of the Pokemon
        text = f"Pokemon {row['name']} (ID: {row['id']}) is a {row['types']} type Pokemon. "
        text += f"It has the following abilities: {row['abilities']}. "
        text += f"Its base stats are: HP: {row['hp']}, Attack: {row['attack']}, Defense: {row['defense']}, "
        text += f"Special Attack: {row['special-attack']}, Special Defense: {row['special-defense']}, Speed: {row['speed']}. "
        text += f"It weighs {row['weight']} units and is {row['height']} units tall."
        
        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "id": row["id"],
                "name": row["name"],
                "types": row["types"],
                "abilities": row["abilities"]
            }
        )
        documents.append(doc)
    
    # Embed documents
    st.session_state.rag_system.embed_documents(documents)
    st.session_state.pokemon_data_embedded = True

# Page config
st.set_page_config(
    page_title="RAG Question Answering",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Question Answering System")
st.markdown("""
This application uses a RAG (Retrieval-Augmented Generation) system to answer questions
based on the provided context. The system uses Google's Gemini model for generation
and ChromaDB for document storage and retrieval.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model settings
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # Document upload
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
    
    if uploaded_file:
        # Save uploaded file
        save_path = Path("data/uploads") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Load and embed document
        with st.spinner("Processing document..."):
            docs = st.session_state.rag_system.load_documents([str(save_path)])
            st.session_state.rag_system.embed_documents(docs)
            st.success("Document processed successfully!")

# Main content
st.header("Ask a Question")

# Question input
question = st.text_input("Enter your question:")

if question:
    # Get answer
    with st.spinner("Generating answer..."):
        result = st.session_state.rag_system.query(question)
        
        # Display answer
        st.subheader("Answer")
        st.write(result["answer"])
        
        # Display context
        with st.expander("View Retrieved Context"):
            for i, ctx in enumerate(result["context"], 1):
                st.markdown(f"**Context {i}:**")
                st.write(ctx)
                st.markdown("---")
        
        # Evaluate response
        if "reference_answer" in st.session_state:
            scores = st.session_state.evaluator.evaluate_response(
                result["answer"],
                st.session_state.reference_answer,
                result["context"]
            )
            
            # Display scores
            st.subheader("Evaluation Scores")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Exact Match", f"{scores['exact_match']:.2f}")
            with col2:
                st.metric("F1 Score", f"{scores['f1_score']:.2f}")
            with col3:
                st.metric("Faithfulness", f"{scores['faithfulness']:.2f}")

# Evaluation section
st.header("Evaluation")
with st.expander("Add Reference Answer"):
    reference = st.text_area("Enter the reference answer:")
    if st.button("Save Reference"):
        st.session_state.reference_answer = reference
        st.success("Reference answer saved!")

# Log hallucinations
if "answer" in locals() and "reference_answer" in st.session_state:
    if scores["faithfulness"] < 0.7:  # Threshold for hallucination
        log_path = Path("hallucinations.csv")
        log_df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "question": question,
            "prediction": result["answer"],
            "reference": st.session_state.reference_answer,
            "faithfulness_score": scores["faithfulness"]
        }])
        
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False) 