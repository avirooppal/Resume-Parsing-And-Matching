import logging

logging.basicConfig(level=logging.DEBUG)

import os
import json
import argparse
import torch
from config.config import config
from utils.file_handler import read_file, load_resume, load_job_description, save_match_results
from utils.preprocessing import preprocess_text, extract_sections
from utils.section_entity_extraction import extract_section_entities, normalize_section_entities
from utils.embedding_matching import load_embedding_model, load_faiss_index, match_resume_to_jd, calculate_embedding_similarity
from utils.feedback_learning import capture_feedback, update_model_with_feedback
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from utils.reranking import load_reranking_model, rerank_matches
from typing import Dict, Any, List
from utils.llm_entity_extraction import extract_entities_with_llm
from utils.pdf_processor import extract_text_from_pdf
from utils.job_description_parser import parse_job_description
from utils.match_scoring import calculate_match_score
from utils.models import load_models
from utils.resume_parser import parse_resume

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")

# Initialize the embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize the reranking model
reranking_model, reranking_tokenizer = load_reranking_model()

logger = logging.getLogger(__name__)

def parse_resume(resume_text: str, ner_model) -> Dict[str, Any]:
    """Parse resume text and extract structured information.
    
    Args:
        resume_text: Raw text from the resume
        ner_model: NER model for entity extraction
        
    Returns:
        Dictionary containing parsed resume information
    """
    # Extract section entities
    section_entities = extract_section_entities(resume_text, ner_model)
    
    # Normalize section entities
    normalized_entities = normalize_section_entities(section_entities)
    
    return normalized_entities

def main():
    """Main function to run the resume matching pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Resume Matching Pipeline')
    parser.add_argument('--resume', required=True, help='Path to resume PDF file')
    parser.add_argument('--jd', required=True, help='Path to job description text file')
    args = parser.parse_args()
    
    # Load models
    models = load_models()
    
    # Process resume
    resume_text = extract_text_from_pdf(args.resume)
    if not resume_text:
        print("Error: Could not extract text from resume PDF")
        return
    
    # Parse resume
    parsed_resume = extract_section_entities(resume_text, models["ner_model"])
    
    # Process job description
    jd_text = load_job_description(args.jd)
    if not jd_text:
        print("Error: Could not load job description")
        return
    
    # Parse job description
    parsed_jd = parse_job_description(jd_text)
    logger.debug(f"Main - Parsed Job Description (parsed_jd): {parsed_jd}")
    
    # Calculate match scores
    match_results = calculate_match_score(parsed_resume, parsed_jd, models["embedding_model"])
    
    # Format results
    results = {
        "resume": parsed_resume,
        "job_description": {
            "title": parsed_jd.get("title", ""),
            "required_skills": parsed_jd.get("required_skills", []),
            "required_experience_years": parsed_jd.get("required_experience_years", 0),
            "required_education": parsed_jd.get("required_education", ""),
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "summary": "",
            "skills": [],
            "work": [],
            "education": [],
            "certifications": [],
            "languages": [],
            "projects": []
        },
        "match_score": match_results
    }
    
    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    resume_name = os.path.splitext(os.path.basename(args.resume))[0]
    jd_name = os.path.splitext(os.path.basename(args.jd))[0]
    output_file = os.path.join(output_dir, f"{resume_name}_vs_{jd_name}_match.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print match results
    print("\nMatch Results:")
    print(f"Overall Score: {match_results['overall_score']:.2f}")
    
    print("\nSkill Matches:")
    print("Matched Skills:", ", ".join(match_results['details']['skill_matches']['matched']))
    print("Missing Skills:", ", ".join(match_results['details']['skill_matches']['missing']))
    print("Semantically Matched Skills:", ", ".join(match_results['details']['skill_matches']['semantically_matched']))
    
    print("\nEducation Matches:")
    print("Matched Education:", ", ".join(match_results['details']['education_matches']['matched']))
    print("Missing Education:", ", ".join(match_results['details']['education_matches']['missing']))
    
    print("\nExperience Matches:")
    print("Matched Experience:", ", ".join(match_results['details']['experience_matches']['matched']))
    print("Missing Experience:", ", ".join(match_results['details']['experience_matches']['missing']))
    
    if 'calculated_experience_years' in match_results:
        print(f"\nCalculated Experience: {match_results['calculated_experience_years']:.1f} years")
    
    if 'semantic_score' in match_results:
        print(f"Semantic Score: {match_results['semantic_score']:.2f}")
    
    if 'skill_score' in match_results:
        print(f"Skill Score: {match_results['skill_score']:.2f}")
    
    if 'experience_score' in match_results:
        print(f"Experience Score: {match_results['experience_score']:.2f}")
    
    if 'education_score' in match_results:
        print(f"Education Score: {match_results['education_score']:.2f}")

if __name__ == "__main__":
    main()

def calculate_skill_match(resume_skills, jd_skills, model):
    matched_skills = []
    missing_skills = []
    semantically_matched_skills = []
    
    resume_skill_names = [skill['name'] for skill in resume_skills]
    
    for skill in jd_skills:
        if skill in resume_skill_names:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)
            for resume_skill in resume_skill_names:
                semantic_score = calculate_semantic_similarity(skill, resume_skill, model)
                if semantic_score > 0.8:  # Threshold for semantic match
                    semantically_matched_skills.append((skill, resume_skill, semantic_score))
    
    return matched_skills, missing_skills, semantically_matched_skills

def calculate_semantic_similarity(text1, text2, model):
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]
    return torch.dot(torch.tensor(embedding1), torch.tensor(embedding2)).item()