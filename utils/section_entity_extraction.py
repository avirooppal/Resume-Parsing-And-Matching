# In utils/section_entity_extraction.py

import re
import flair
import os
import logging
from flair.data import Sentence
from flair.models import SequenceTagger
from dateparser import parse
from config.config import config
from utils.file_handler import load_skills_ontology, load_job_title_mapping
from utils.preprocessing import preprocess_text
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_ner_model():
    model_path = config['ner_model_path']
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Download and save the model
        model = SequenceTagger.load('ner')
        model.save(model_path)
    return SequenceTagger.load(model_path)

def extract_dates(text):
    dates = []
    lines = text.split('\n')
    for line in lines:
        date = parse(line)
        if date:
            dates.append(date)
    return dates

def extract_skills(text: str) -> List[Dict[str, str]]:
    """Extract skills from text.
    
    Args:
        text: Skills section text
        
    Returns:
        List of dictionaries with 'name' and 'level' keys
    """
    # Split by common delimiters
    skills = re.split(r'[,•|\n]', text)
    
    # Clean and filter skills
    cleaned_skills = []
    for skill in skills:
        skill = skill.strip()
        # Remove leading bullet points or dashes
        skill = re.sub(r'^[-•]\s*', '', skill)
        if skill:
            cleaned_skills.append(skill)
    
    # Convert to list of dictionaries
    return [{"name": skill, "level": "Intermediate"} for skill in cleaned_skills]

def extract_roles(text):
    job_title_mapping = load_job_title_mapping()
    roles = []
    for role in job_title_mapping:
        if role in text:
            roles.append(job_title_mapping[role])
    return roles

def extract_section_entities(text: str, ner_model: SequenceTagger) -> Dict[str, Any]:
    """Extract entities from resume text using NER model.
    
    Args:
        text: Resume text
        ner_model: NER model for entity extraction
        
    Returns:
        Dictionary containing extracted entities
    """
    # Split text into sections
    sections = split_into_sections(text)
    logger.info(f"Split text into sections: {list(sections.keys())}")
    
    # Initialize entities dictionary
    entities = {
        'name': '',
        'email': '',
        'phone': '',
        'location': '',
        'summary': '',
        'skills': [],
        'work': [],
        'education': [],
        'certifications': [],
        'languages': [],
        'projects': []
    }
    
    # Process each section
    for section_name, section_text in sections.items():
        logger.info(f"Processing section: {section_name}")
        logger.info(f"Section text: {section_text[:200]}...")  # Log first 200 chars
        
        if section_name == 'summary':
            entities['summary'] = section_text
        elif section_name == 'skills':
            skills = extract_skills(section_text)
            logger.info(f"Extracted skills: {skills}")
            entities['skills'] = skills
        elif section_name == 'work':
            work = extract_work_experience(section_text)
            logger.info(f"Extracted work experience: {work}")
            entities['work'] = work
        elif section_name == 'education':
            education = extract_education(section_text)
            logger.info(f"Extracted education: {education}")
            entities['education'] = education
        elif section_name == 'certifications':
            entities['certifications'] = extract_certifications(section_text)
        elif section_name == 'languages':
            entities['languages'] = extract_languages(section_text)
        elif section_name == 'projects':
            entities['projects'] = extract_projects(section_text)
    
    # Extract contact information from the very top of the resume, before any sections
    first_section_start_index = len(text) # Default to end of text if no sections
    # Find the start of the very first recognized section to get the header part
    first_header_match = re.search(r'(?i)(?:summary|profile|objective|skills|experience|work|education|certifications|languages|projects)[\s:–-]*\n', text)
    if first_header_match:
        first_section_start_index = first_header_match.start()
    
    text_for_contact_info = text[:first_section_start_index].strip()
    logger.debug(f"extract_section_entities - Text for contact info: {text_for_contact_info[:200]}...")

    contact_info = extract_contact_info(text_for_contact_info, ner_model) # Pass ner_model here
    logger.info(f"Extracted contact info: {contact_info}")
    entities.update(contact_info)
    
    return entities

def split_into_sections(text: str) -> Dict[str, str]:
    """Split resume text into sections using more robust regex."""
    section_headers = [
        "summary", "profile", "objective", "skills", "technologies", "expertise", "competencies", "technical skills",
        "experience", "work experience", "employment", "professional experience",
        "education", "academic", "qualification", "certifications", "certificates", "accreditations",
        "languages", "language proficiency", "projects", "project experience", "portfolio"
    ]
    pattern = r'(?P<header>' + '|'.join([re.escape(h) for h in section_headers]) + r')[\s:–-]*\n'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    sections = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = match.group('header').lower()
        canonical = (
            "summary" if "summary" in header or "profile" in header or "objective" in header else
            "skills" if "skill" in header or "technolog" in header or "expertise" in header or "competenc" in header else
            "work" if "experience" in header or "employment" in header else
            "education" if "education" in header or "academic" in header or "qualification" in header else
            "certifications" if "certificat" in header or "accreditation" in header else
            "languages" if "language" in header else
            "projects" if "project" in header or "portfolio" in header else
            header
        )
        sections[canonical] = text[start:end].strip()
    return sections

def extract_location(text: str) -> str:
    loc_match = re.search(r'Location:\s*(.+)', text, re.IGNORECASE)
    if loc_match:
        loc = loc_match.group(1).strip()
        if loc.upper() != 'N/A':
            return loc
    return ""

def extract_contact_info(text: str, ner_model: SequenceTagger) -> Dict[str, str]:
    """Extract contact information from resume text using Flair NER with fallback logic."""
    logger.debug(f"extract_contact_info - Processing text: {text[:100]}...")

    name = ""
    location_parts = []

    # Regex-based extraction first
    email = extract_email(text)
    phone = extract_phone(text)
    location = extract_location(text)

    try:
        sentence = Sentence(text)
        ner_model.predict(sentence)
        for entity in sentence.get_spans('ner'):
            logger.debug(f"extract_contact_info - Found entity: {entity.text} (Tag: {entity.tag}, Score: {entity.score:.2f})")
            if entity.tag == 'PER':
                if not name:
                    name = entity.text
            elif entity.tag == 'LOC':
                if entity.text.upper() != 'N/A':
                    location_parts.append(entity.text)
    except Exception as e:
        logger.error(f"Error in NER model prediction: {e}")

    # Fallback logic for name
    if not name:
        logger.warning("NER did not identify a person's name. Applying heuristic fallbacks.")
        lines = text.split('\n')
        # Heuristic A: Try to extract name using "Name:" prefix
        for line in lines:
            line = line.strip()
            if not line:
                continue
            name_match = re.match(r'Name:\s*(.+)', line, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                if len(potential_name.split()) <= 3:
                    name = potential_name
                    logger.debug(f"Heuristic fallback set name via 'Name:' prefix: '{name}'")
                    break
        # Heuristic B: Try to extract name based on capitalization
        if not name:
            for line in lines:
                line = line.strip()
                if not line or '@' in line:
                    continue
                parts = line.split()
                if 1 < len(parts) <= 3 and all(part and part[0].isupper() for part in parts):
                    name = line
                    logger.debug(f"Heuristic fallback set name via capitalization: '{name}'")
                    break

    final_location = ", ".join([loc for loc in location_parts if loc.upper() != 'N/A'])

    return {
        'name': name,
        'email': email,
        'phone': phone,
        'location': final_location or location
    }

def extract_work_experience(text: str) -> List[Dict[str, Any]]:
    """Extract work experience entries with improved parsing."""
    work_experiences = []
    current_experience = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for company line (contains company name and dates)
        if '|' in line:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 3:
                position = parts[0]
                company = parts[1]
                dates = parts[2]
                
                # Extract dates
                date_match = re.search(r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4}|Present)', dates)
                if date_match:
                    start_date = date_match.group(1)
                    end_date = date_match.group(2)
                    if end_date == "Present":
                        end_date = datetime.now().strftime("%Y")
                else:
                    # Try to find just years
                    years = re.findall(r'\d{4}', dates)
                    if len(years) >= 2:
                        start_date = years[0]
                        end_date = years[1]
                    else:
                        start_date = ""
                        end_date = ""
                
                if current_experience:
                    work_experiences.append(current_experience)
                
                current_experience = {
                    "company": company,
                    "position": position,
                    "startDate": start_date,
                    "endDate": end_date,
                    "summary": ""
                }
        elif current_experience:
            # Add to summary
            if current_experience["summary"]:
                current_experience["summary"] += " " + line
            else:
                current_experience["summary"] = line
    
    if current_experience:
        work_experiences.append(current_experience)
    
    return work_experiences

def extract_education(text: str) -> List[Dict[str, Any]]:
    """Extract education entries with improved parsing."""
    education_entries = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for education line (contains degree, institution, and dates)
        if '|' in line:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 3:
                degree = parts[0]
                institution = parts[1]
                dates = parts[2]
                
                # Extract dates
                years = re.findall(r'\d{4}', dates)
                if len(years) >= 2:
                    start_date = years[0]
                    end_date = years[1]
                else:
                    start_date = ""
                    end_date = ""
                
                education_entries.append({
                    "studyType": degree,
                    "institution": institution,
                    "startDate": start_date,
                    "endDate": end_date,
                    "area": ""  # Could be extracted if available
                })
    
    return education_entries

def extract_certifications(text: str) -> List[str]:
    """Extract certifications from text.
    
    Args:
        text: Certifications section text
        
    Returns:
        List of certifications
    """
    # Split by common delimiters
    certs = re.split(r'[,•|\n]', text)
    
    # Clean and filter certifications
    certs = [cert.strip() for cert in certs if cert.strip()]
    
    return certs

def extract_languages(text: str) -> List[str]:
    """Extract languages from text.
    
    Args:
        text: Languages section text
        
    Returns:
        List of languages
    """
    # Split by common delimiters
    languages = re.split(r'[,•|\n]', text)
    
    # Clean and filter languages
    languages = [lang.strip() for lang in languages if lang.strip()]
    
    return languages

def extract_projects(text: str) -> List[Dict[str, Any]]:
    """Extract projects from text.
    
    Args:
        text: Projects section text
        
    Returns:
        List of project entries
    """
    # Split into individual projects
    projects = re.split(r'\n\s*\n', text)
    
    project_entries = []
    for project in projects:
        if not project.strip():
            continue
            
        # Extract project name and dates
        name_match = re.search(r'(.*?)(?:\s*[-–]\s*|\n)(.*?)(?:\s*[-–]\s*|\n)', project)
        if name_match:
            name = name_match.group(1).strip()
            dates = name_match.group(2).strip()
            
            # Extract description
            desc_match = re.search(r'(?:\n)(.*)', project, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ''
            
            project_entries.append({
                'name': name,
                'dates': dates,
                'description': description
            })
    
    return project_entries

def extract_email(text: str) -> str:
    """Extract email address using regex pattern."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else ""

def extract_phone(text: str) -> str:
    """Extract phone number using regex pattern."""
    phone_pattern = r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else ""

def normalize_section_entities(section_entities: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize section entities to ensure consistent format."""
    normalized = section_entities.copy()
    
    # Ensure skills are in the correct format
    if "skills" in normalized:
        if isinstance(normalized["skills"], list):
            normalized["skills"] = [
                {"name": skill["name"], "level": skill.get("level", "Intermediate")}
                if isinstance(skill, dict)
                else {"name": str(skill), "level": "Intermediate"}
                for skill in normalized["skills"]
            ]
    
    # Ensure work experience is in the correct format
    if "work" in normalized:
        if isinstance(normalized["work"], list):
            normalized["work"] = [
                {
                    "company": exp.get("company", ""),
                    "position": exp.get("position", ""),
                    "startDate": exp.get("startDate", ""),
                    "endDate": exp.get("endDate", ""),
                    "summary": exp.get("summary", "")
                }
                for exp in normalized["work"]
            ]
    
    # Ensure education is in the correct format
    if "education" in normalized:
        if isinstance(normalized["education"], list):
            normalized["education"] = [
                {
                    "studyType": edu.get("studyType", ""),
                    "area": edu.get("area", ""),
                    "institution": edu.get("institution", ""),
                    "startDate": edu.get("startDate", ""),
                    "endDate": edu.get("endDate", "")
                }
                for edu in normalized["education"]
            ]
    
    return normalized