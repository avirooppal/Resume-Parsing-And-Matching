from config.config import config
from utils.file_handler import load_feedback_data, save_feedback_data
import json
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

def capture_feedback(match_results: Dict[str, Any], feedback: Dict[str, Any]) -> None:
    """Capture user feedback on match results."""
    feedback_data = {
        "match_results": match_results,
        "feedback": feedback,
        "timestamp": str(datetime.now())
    }
    
    # Load existing feedback
    try:
        with open(config["feedback_data_path"], 'r') as f:
            existing_feedback = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_feedback = []
    
    # Append new feedback
    existing_feedback.append(feedback_data)
    
    # Save updated feedback
    with open(config["feedback_data_path"], 'w') as f:
        json.dump(existing_feedback, f, indent=2)

def update_model_with_feedback(model: SentenceTransformer, feedback_data: List[Dict[str, Any]]) -> SentenceTransformer:
    """Update the model based on user feedback."""
    # Prepare training data from feedback
    train_examples = []
    
    for feedback in feedback_data:
        match_results = feedback["match_results"]
        user_feedback = feedback["feedback"]
        
        # Extract relevant text
        resume_text = []
        if match_results["resume"].get("basics", {}).get("summary"):
            resume_text.append(match_results["resume"]["basics"]["summary"])
        for work in match_results["resume"].get("work", []):
            if work.get("summary"):
                resume_text.append(work["summary"])
        
        jd_text = []
        if match_results["job_description"].get("description"):
            jd_text.append(match_results["job_description"]["description"])
        for req in match_results["job_description"].get("requirements", []):
            jd_text.append(req)
        
        # Combine texts
        resume_combined = " ".join(resume_text)
        jd_combined = " ".join(jd_text)
        
        # Create training example
        if user_feedback.get("is_good_match", False):
            # Positive example
            train_examples.append((resume_combined, jd_combined, 1.0))
        else:
            # Negative example
            train_examples.append((resume_combined, jd_combined, 0.0))
    
    if train_examples:
        # Fine-tune model on feedback data
        model.fit(
            train_examples,
            epochs=1,
            warmup_steps=100,
            show_progress_bar=True
        )
    
    return model

def load_feedback_data() -> List[Dict[str, Any]]:
    """Load feedback data from file."""
    try:
        with open(config["feedback_data_path"], 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_feedback_data(feedback_data):
    with open(config['feedback_data_path'], 'w') as file:
        json.dump(feedback_data, file, indent=4)

def retrain_models(feedback_data):
    # Placeholder for retraining logic
    pass