import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import read_index
from config.config import config
from typing import Dict, Any, List, Tuple
import os

def load_embedding_model():
    return SentenceTransformer(config['embedding_model_path'])

def load_faiss_index():
    try:
        index_path = os.path.join(config["models_dir"], "faiss_index.bin")
        if os.path.exists(index_path):
            return read_index(index_path)
        return None
    except ImportError:
        print("FAISS not installed. Please install it using: pip install faiss-cpu")
        return None

def embed_text(text, model):
    return model.encode([text])[0]

def search_similar_texts(query_embedding, faiss_index, top_k=5):
    D, I = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    return D[0], I[0]

def calculate_semantic_similarity(text1, text2, model):
    embedding1 = embed_text(text1, model)
    embedding2 = embed_text(text2, model)
    return torch.dot(torch.tensor(embedding1), torch.tensor(embedding2)).item()

def hybrid_scoring(semantic_score, skill_score, experience_score, education_score, weights):
    return (semantic_score * weights['semantic'] +
            skill_score * weights['skills'] +
            experience_score * weights['experience'] +
            education_score * weights['education'])

def calculate_embedding_similarity(resume: Dict[str, Any], job_description: Dict[str, Any], model: SentenceTransformer) -> float:
    """Calculate semantic similarity between resume and job description using embeddings."""
    # Extract relevant text from resume
    resume_text = []
    if resume.get("basics", {}).get("summary"):
        resume_text.append(resume["basics"]["summary"])
    for work in resume.get("work", []):
        if work.get("summary"):
            resume_text.append(work["summary"])
    for skill in resume.get("skills", []):
        if skill.get("name"):
            resume_text.append(skill["name"])
    
    # Extract relevant text from job description
    jd_text = []
    if job_description.get("description"):
        jd_text.append(job_description["description"])
    for req in job_description.get("requirements", []):
        jd_text.append(req)
    for skill in job_description.get("skills", []):
        jd_text.append(skill)
    
    # Combine texts
    resume_combined = " ".join(resume_text)
    jd_combined = " ".join(jd_text)
    
    # Calculate embeddings
    resume_embedding = model.encode([resume_combined])[0]
    jd_embedding = model.encode([jd_combined])[0]
    
    # Calculate cosine similarity
    similarity = np.dot(resume_embedding, jd_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
    )
    
    return float(similarity)

def match_resume_to_jd(resume_text: str, jd_text: str, model: SentenceTransformer) -> float:
    """Calculate semantic similarity between resume and job description texts."""
    # Calculate embeddings
    resume_embedding = model.encode([resume_text])[0]
    jd_embedding = model.encode([jd_text])[0]
    
    # Calculate cosine similarity
    similarity = np.dot(resume_embedding, jd_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
    )
    
    return float(similarity)