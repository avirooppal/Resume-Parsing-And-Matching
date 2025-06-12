import fitz  # PyMuPDF
from typing import Dict, Any, List
from utils.section_entity_extraction import extract_section_entities
from utils.skill_role_normalization import normalize_skills, normalize_roles

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract text from each page
        for page in doc:
            text += page.get_text()
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

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
    
    # Normalize skills and roles
    if 'skills' in section_entities:
        section_entities['skills'] = normalize_skills(section_entities['skills'])
    if 'roles' in section_entities:
        section_entities['roles'] = normalize_roles(section_entities['roles'])
    
    # Structure the resume data
    resume_data = {
        "basics": {
            "name": section_entities.get('name', ''),
            "email": section_entities.get('email', ''),
            "phone": section_entities.get('phone', ''),
            "location": section_entities.get('location', ''),
            "summary": section_entities.get('summary', '')
        },
        "skills": section_entities.get('skills', []),
        "work": section_entities.get('work', []),
        "education": section_entities.get('education', []),
        "certifications": section_entities.get('certifications', []),
        "languages": section_entities.get('languages', []),
        "projects": section_entities.get('projects', [])
    }
    
    return resume_data

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract contact information from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary containing contact information
    """
    # TODO: Implement contact information extraction
    return {
        "name": "",
        "email": "",
        "phone": "",
        "location": ""
    }

def extract_work_experience(text: str) -> List[Dict[str, Any]]:
    """Extract work experience from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        List of work experience entries
    """
    # TODO: Implement work experience extraction
    return []

def extract_education(text: str) -> List[Dict[str, Any]]:
    """Extract education information from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        List of education entries
    """
    # TODO: Implement education extraction
    return []

def extract_skills(text: str) -> List[str]:
    """Extract skills from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        List of skills
    """
    # TODO: Implement skills extraction
    return [] 