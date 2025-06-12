import json
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESUMES_DIR = os.path.join(BASE_DIR, "resumes")
FILES_DIR = os.path.join(BASE_DIR, "files")

# Model paths
NER_MODEL_PATH = os.path.join(MODELS_DIR, "ner_model")
RERANKING_MODEL_PATH = os.path.join(MODELS_DIR, "reranking_model")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "embedding_model")

# Data paths
SKILLS_ONTOLOGY_PATH = os.path.join(DATA_DIR, "skills_ontology.json")
JOB_TITLE_MAPPING_PATH = os.path.join(DATA_DIR, "job_title_mapping.json")

# File paths
RESUME_PATH = os.path.join(RESUMES_DIR, "Ravi_Sharma_Resume.pdf")
JOB_DESCRIPTION_PATH = os.path.join(FILES_DIR, "Job_Description_1.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "match_results.json")

# Model configuration
MODEL_CONFIG = {
    "sentence_transformer": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_seq_length": 128
    },
    "reranking_model": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_seq_length": 512
    },
    "ner_model": {
        "model_name": "flair/ner-english",
        "max_seq_length": 256
    }
}

# Matching configuration
MATCHING_CONFIG = {
    "skill_match_threshold": 0.7,
    "education_match_threshold": 0.8,
    "experience_match_threshold": 0.8,
    "embedding_similarity_threshold": 0.6
}

# Configuration dictionary
config = {
    "base_dir": BASE_DIR,
    "models_dir": MODELS_DIR,
    "data_dir": DATA_DIR,
    "resume_dir": RESUMES_DIR,
    "job_description_dir": FILES_DIR,
    "ner_model_path": NER_MODEL_PATH,
    "reranking_model_path": RERANKING_MODEL_PATH,
    "embedding_model_path": EMBEDDING_MODEL_PATH,
    "skills_ontology_path": SKILLS_ONTOLOGY_PATH,
    "job_title_mapping_path": JOB_TITLE_MAPPING_PATH,
    "resume_path": RESUME_PATH,
    "job_description_path": JOB_DESCRIPTION_PATH,
    "output_path": OUTPUT_PATH,
    "model_config": MODEL_CONFIG,
    "matching_config": MATCHING_CONFIG
} 