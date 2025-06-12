from config import config
from utils.file_handler import load_skills_ontology, load_job_title_mapping
from utils.preprocessing import preprocess_text
import json
from typing import Dict, Any, List
from symspellpy import SymSpell, Verbosity

def normalize_skills(skills):
    skills_ontology = load_skills_ontology()
    normalized_skills = []
    for skill in skills:
        if skill in skills_ontology['skills']:
            normalized_skills.append(skills_ontology['skills'][skill])
        else:
            normalized_skills.append(skill)
    return normalized_skills

def normalize_roles(roles):
    job_title_mapping = load_job_title_mapping()
    normalized_roles = []
    for role in roles:
        if role in job_title_mapping:
            normalized_roles.append(job_title_mapping[role])
        else:
            normalized_roles.append(role)
    return normalized_roles

def normalize_section_entities(section_entities):
    for section, content in section_entities.items():
        content['skills'] = normalize_skills(content['skills'])
        content['roles'] = normalize_roles(content['roles'])
    return section_entities

def load_skills_ontology() -> Dict[str, Any]:
    """Load skills ontology from JSON file."""
    try:
        with open(config["skills_ontology_path"], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading skills ontology: {str(e)}")
        return {}

def load_job_title_mapping() -> Dict[str, str]:
    """Load job title mapping from JSON file."""
    try:
        with open(config["job_title_mapping_path"], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading job title mapping: {str(e)}")
        return {}

def normalize_skill(skill: str) -> str:
    """Normalize a skill name using the skills ontology."""
    # Load skills ontology
    skills_ontology = load_skills_ontology()
    
    # Convert to lowercase for case-insensitive matching
    skill_lower = skill.lower().strip()
    
    # Check for exact match
    if skill_lower in skills_ontology:
        return skills_ontology[skill_lower]["normalized_name"]
    
    # Check for aliases
    for normalized_name, data in skills_ontology.items():
        if skill_lower in data.get("aliases", []):
            return data["normalized_name"]
    
    # If no match found, return the original skill
    return skill

def normalize_role(role: str) -> str:
    """Normalize a job role using the job title mapping."""
    # Load job title mapping
    job_title_mapping = load_job_title_mapping()
    
    # Convert to lowercase for case-insensitive matching
    role_lower = role.lower().strip()
    
    # Check for exact match
    if role_lower in job_title_mapping:
        return job_title_mapping[role_lower]
    
    # Check for partial matches
    for original, normalized in job_title_mapping.items():
        if role_lower in original.lower() or original.lower() in role_lower:
            return normalized
    
    # If no match found, return the original role
    return role

def create_symspell_dictionary() -> SymSpell:
    """Create a SymSpell dictionary for fuzzy matching."""
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Load skills ontology
    skills_ontology = load_skills_ontology()
    
    # Add skills to dictionary
    for skill in skills_ontology:
        sym_spell.create_dictionary_entry(skill, 1)
        for alias in skills_ontology[skill].get("aliases", []):
            sym_spell.create_dictionary_entry(alias, 1)
    
    return sym_spell

def fuzzy_match_skill(skill: str, sym_spell: SymSpell) -> str:
    """Fuzzy match a skill using SymSpell."""
    # Look up the skill
    suggestions = sym_spell.lookup(skill, Verbosity.CLOSEST, max_edit_distance=2)
    
    if suggestions:
        # Return the best match
        return suggestions[0].term
    
    # If no match found, return the original skill
    return skill