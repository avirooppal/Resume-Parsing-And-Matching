import os
import faiss
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from config.config import config

def setup_directories():
    # Create necessary directories
    directories = [
        'models',
        'data',
        'files'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download and save embedding model
    print("Downloading embedding model...")
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding_model.save(config['embedding_model_path'])
    
    # Download and save reranking model
    print("Downloading reranking model...")
    reranking_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_model = AutoModelForSequenceClassification.from_pretrained(reranking_model_name)
    reranking_tokenizer = AutoTokenizer.from_pretrained(reranking_model_name)
    
    reranking_model.save_pretrained(config['reranking_model_path'])
    reranking_tokenizer.save_pretrained(config['reranking_model_path'])

def create_faiss_index():
    print("Creating FAISS index...")
    # Initialize the embedding model
    model = SentenceTransformer(config['embedding_model_path'])
    
    # Create a simple FAISS index
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    
    # Save the index
    faiss.write_index(index, config['faiss_index_path'])
    print(f"FAISS index created and saved to {config['faiss_index_path']}")

def main():
    download_models()
    create_faiss_index()
    print("Setup completed successfully!")

if __name__ == "__main__":
    print("Setting up directories...")
    setup_directories()
    print("Downloading models...")
    main()
    print("Setup complete!")