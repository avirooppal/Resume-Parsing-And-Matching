import os
import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flair.models import SequenceTagger
from config.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models() -> Dict[str, Any]:
    """Load all required models for the pipeline.
    
    Returns:
        Dictionary containing loaded models
    """
    models = {}
    
    # Load sentence transformer model for embeddings
    try:
        logger.info("Loading sentence transformer model...")
        model_name = config["model_config"]["sentence_transformer"]["model_name"]
        logger.info(f"Using model: {model_name}")
        
        # Create models directory if it doesn't exist
        os.makedirs(config["models_dir"], exist_ok=True)
        
        # Load the model
        embedding_model = SentenceTransformer(model_name)
        logger.info("Successfully loaded sentence transformer model")
        
        # Store the model
        models["embedding_model"] = embedding_model
        logger.info("Stored embedding model in models dictionary")
        
    except Exception as e:
        logger.error(f"Error loading sentence transformer model: {str(e)}")
        logger.error(f"Model name: {model_name}")
        logger.error(f"Config: {config['model_config']['sentence_transformer']}")
        raise
    
    # Load reranking model
    try:
        logger.info("Loading reranking model...")
        model_name = config["model_config"]["reranking_model"]["model_name"]
        logger.info(f"Using model: {model_name}")
        
        # Load model and tokenizer
        reranking_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # For regression task
        )
        reranking_tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Successfully loaded reranking model and tokenizer")
        
        # Store the models
        models["reranking_model"] = reranking_model
        models["reranking_tokenizer"] = reranking_tokenizer
        logger.info("Stored reranking model and tokenizer in models dictionary")
        
    except Exception as e:
        logger.error(f"Error loading reranking model: {str(e)}")
        logger.error(f"Model name: {model_name}")
        logger.error(f"Config: {config['model_config']['reranking_model']}")
        raise
    
    # Load NER model
    try:
        logger.info("Loading NER model...")
        model_name = config["model_config"]["ner_model"]["model_name"]
        logger.info(f"Using model: {model_name}")
        
        # Load the model
        ner_model = SequenceTagger.load(model_name)
        logger.info("Successfully loaded NER model")
        
        # Store the model
        models["ner_model"] = ner_model
        logger.info("Stored NER model in models dictionary")
        
    except Exception as e:
        logger.error(f"Error loading NER model: {str(e)}")
        logger.error(f"Model name: {model_name}")
        logger.error(f"Config: {config['model_config']['ner_model']}")
        raise
    
    # Verify all required models are loaded
    required_models = ["embedding_model", "reranking_model", "reranking_tokenizer", "ner_model"]
    missing_models = [model for model in required_models if model not in models]
    
    if missing_models:
        error_msg = f"Failed to load the following models: {', '.join(missing_models)}"
        logger.error(error_msg)
        logger.error(f"Available models: {list(models.keys())}")
        raise RuntimeError(error_msg)
    
    logger.info("All models loaded successfully")
    logger.info(f"Available models: {list(models.keys())}")
    return models 