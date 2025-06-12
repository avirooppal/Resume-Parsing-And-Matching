from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from flair.models import SequenceTagger
import os
from config.config import config

def download_models():
    """Download and save all models."""
    # Download sentence transformer model
    SentenceTransformer('all-MiniLM-L6-v2').save(config["sentence_transformer_path"])

    # Download reranking model
    reranking_model_path = config["reranking_model_path"]
    os.makedirs(os.path.dirname(reranking_model_path), exist_ok=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained('facebook/tiny-lm-1.1B-Chat')
    rerank_tokenizer = AutoTokenizer.from_pretrained('facebook/tiny-lm-1.1B-Chat')
    rerank_model.save_pretrained(reranking_model_path)
    rerank_tokenizer.save_pretrained(reranking_model_path)

    # Download and save Flair NER model
    ner_model_path = config["ner_model_path"]
    os.makedirs(os.path.dirname(ner_model_path), exist_ok=True)
    ner_model = SequenceTagger.load('ner')
    ner_model.save(ner_model_path)

def load_models():
    """Load all pre-downloaded models."""
    models = {}

    # Load sentence transformer
    models["sentence_transformer"] = SentenceTransformer(config["sentence_transformer_path"])

    # Load reranking model
    models["reranking_model"] = AutoModelForSequenceClassification.from_pretrained(config["reranking_model_path"])
    models["reranking_tokenizer"] = AutoTokenizer.from_pretrained(config["reranking_model_path"])

    # Load NER model
    models["ner_model"] = SequenceTagger.load(config["ner_model_path"])

    return models
