# utils/match_scoring.py

import re
import logging
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

# Configure logging for this module
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_total_experience(work_experience: List[Dict[str, Any]]) -> float:
    """Calculate total years of experience from work history more robustly."""
    total_years = 0.0
    for exp in work_experience:
        start_date_str = exp.get("startDate", "")
        end_date_str = exp.get("endDate", "")

        if start_date_str and end_date_str:
            try:
                # Use regex to find any 4-digit year
                start_year_match = re.search(r'\b(19|20)\d{2}\b', start_date_str)
                end_year_match = re.search(r'\b(19|20)\d{2}\b', end_date_str)

                if start_year_match:
                    start_year = int(start_year_match.group(0))
                    # If end date is 'Present' or similar, use the current year
                    end_year = int(end_year_match.group(0)) if end_year_match else datetime.now().year
                    
                    # Basic calculation of duration
                    duration = end_year - start_year
                    # A more precise calculation could parse months, but years are often sufficient
                    
                    total_years += duration if duration > 0 else 1 # Assume at least one year for short stints

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse dates to calculate experience: {start_date_str} - {end_date_str}. Error: {e}")
                continue
    
    logger.debug(f"calculate_total_experience - Total calculated years: {total_years}")
    return total_years

def _calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Helper to calculate semantic similarity between two texts."""
    try:
        embedding1 = model.encode([text1], convert_to_tensor=True)
        embedding2 = model.encode([text2], convert_to_tensor=True)
        # Use PyTorch utilities for cosine similarity for robustness
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return cos_sim.item()
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_match_score(resume_data: Dict[str, Any], jd_data: Dict[str, Any], embedding_model: SentenceTransformer) -> Dict[str, Any]:
    """
    Calculates a comprehensive match score between a resume and a job description.

    This is the primary function that orchestrates the scoring by comparing skills,
    experience, and education, and returns a structured dictionary with all details.
    """
    logger.debug(f"Starting match score calculation...")

    # --- 1. Skill Matching ---
    resume_skills_list = resume_data.get("skills", [])
    jd_skills = jd_data.get("required_skills", [])
    resume_skill_names = [skill.get("name", "").lower() for skill in resume_skills_list]
    
    matched_skills = []
    missing_skills = []
    semantically_matched = []
    
    # Use a copy to safely remove items
    unmatched_jd_skills = list(jd_skills)

    # Exact matches first
    for resume_skill in resume_skill_names:
        for jd_skill in list(unmatched_jd_skills):
            if resume_skill == jd_skill.lower():
                matched_skills.append(jd_skill)
                unmatched_jd_skills.remove(jd_skill)

    # Semantic matches for the remaining
    for jd_skill in list(unmatched_jd_skills):
        best_match = (None, 0.0)
        for resume_skill in resume_skill_names:
            similarity = _calculate_semantic_similarity(jd_skill, resume_skill, embedding_model)
            if similarity > best_match[1]:
                best_match = (resume_skill, similarity)
        
        # Use a threshold for semantic matching (e.g., > 0.7)
        if best_match[1] > 0.7:
            semantically_matched.append((jd_skill, best_match[0].title(), best_match[1]))
            unmatched_jd_skills.remove(jd_skill)
    
    missing_skills = unmatched_jd_skills
    
    skill_score = (len(matched_skills) + len(semantically_matched)) / len(jd_skills) if jd_skills else 1.0

    # --- 2. Experience Matching ---
    resume_experience_years = calculate_total_experience(resume_data.get("work", []))
    required_experience_years = jd_data.get("required_experience_years", 0)
    
    experience_score = 1.0
    if required_experience_years > 0:
        experience_score = min(1.0, resume_experience_years / required_experience_years)

    # --- 3. Education Matching ---
    required_education_str = jd_data.get("required_education", "")
    education_score = 1.0  # Default to 1.0 if no education is required
    if required_education_str:
        education_score = 0.0 # Assume no match until found
        # NOTE: This logic is simple. "B.Tech" does not contain "Bachelor".
        # A robust solution needs a degree equivalency map (e.g., B.Tech -> Bachelor's)
        keywords = ["bachelor", "b.tech", "master", "m.tech", "phd"]
        req_edu_lower = required_education_str.lower()
        
        for edu in resume_data.get("education", []):
            study_type_lower = edu.get("studyType", "").lower()
            if any(keyword in req_edu_lower and keyword in study_type_lower for keyword in keywords):
                education_score = 1.0
                break

    # --- 4. Semantic Document Score ---
    resume_full_text = resume_data.get("summary", "") + " ".join(resume_skill_names)
    jd_full_text = jd_data.get("match_text", "")
    semantic_score = _calculate_semantic_similarity(resume_full_text, jd_full_text, embedding_model)

    # --- 5. Final Weighted Score ---
    weights = {"skills": 0.5, "experience": 0.3, "education": 0.1, "semantic": 0.1}
    overall_score = (
        skill_score * weights["skills"] +
        experience_score * weights["experience"] +
        education_score * weights["education"] +
        semantic_score * weights["semantic"]
    )

    # --- 6. Construct Final Result Dictionary ---
    # This structure matches what main.py expects
    results = {
        'overall_score': overall_score,
        'skill_score': skill_score,
        'experience_score': experience_score,
        'education_score': education_score,
        'semantic_score': semantic_score,
        'calculated_experience_years': resume_experience_years,
        'details': {
            'skill_matches': {
                'matched': matched_skills,
                'missing': missing_skills,
                'semantically_matched': [f"{jd} (similar to {res}, score: {sim:.2f})" for jd, res, sim in semantically_matched]
            },
            'education_matches': {
                'matched': [required_education_str] if education_score == 1.0 and required_education_str else [],
                'missing': [required_education_str] if education_score < 1.0 and required_education_str else []
            },
            'experience_matches': {
                'matched': [f"Candidate has {resume_experience_years:.1f} years (required: {required_experience_years})"] if experience_score >= 1.0 else [],
                'missing': [f"Candidate has {resume_experience_years:.1f} years (required: {required_experience_years})"] if experience_score < 1.0 else []
            }
        }
    }
    logger.debug(f"Final match results: {results}")
    return results