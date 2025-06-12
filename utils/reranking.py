from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.config import config
import torch
from typing import Dict, Any, List

def load_reranking_model():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_matches_for_reranking(resume_data: Dict[str, Any], jd_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare matches for reranking by combining relevant sections."""
    matches = []
    
    # Combine skills
    resume_skills = resume_data.get("skills", [])
    jd_skills = jd_data.get("skills", [])
    
    # Handle both string and dictionary skill formats
    for skill in resume_skills:
        if isinstance(skill, dict):
            skill_name = skill.get("name", "")
        else:
            skill_name = str(skill)
        
        if skill_name:
            matches.append({
                "type": "skill",
                "text": f"Skill: {skill_name}",
                "original": skill
            })
    
    # Combine work experience
    for work in resume_data.get("work", []):
        if isinstance(work, dict):
            position = work.get("position", "")
            company = work.get("company", "")
            if position or company:
                matches.append({
                    "type": "experience",
                    "text": f"Experience: {position} at {company}",
                    "original": work
                })
    
    # Combine education
    for edu in resume_data.get("education", []):
        if isinstance(edu, dict):
            study_type = edu.get("studyType", "")
            area = edu.get("area", "")
            if study_type or area:
                matches.append({
                    "type": "education",
                    "text": f"Education: {study_type} in {area}",
                    "original": edu
                })
    
    return matches

def rerank_matches(resume_data: Dict[str, Any], jd_data: Dict[str, Any], model) -> List[Dict[str, Any]]:
    """Rerank matches between resume and job description.
    
    Args:
        resume_data: Parsed resume data
        jd_data: Parsed job description data
        model: Reranking model
        
    Returns:
        List of reranked matches
    """
    # Prepare matches for reranking
    matches = prepare_matches_for_reranking(resume_data, jd_data)
    
    if not matches:
        return []
    
    # Get tokenizer from AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Prepare input for the model
    inputs = [tokenizer(match["text"], return_tensors="pt") for match in matches]
    
    # Get scores from the model
    scores = []
    for input in inputs:
        with torch.no_grad():
            outputs = model(**input)
            # Use sigmoid for regression scores
            score = torch.sigmoid(outputs.logits).item()
            scores.append(score)
    
    # Sort matches by score
    ranked_matches = sorted(
        zip(matches, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return ranked matches with scores
    return [
        {
            "match": match,
            "score": score,
            "type": match["type"],
            "text": match["text"]
        }
        for match, score in ranked_matches
    ]