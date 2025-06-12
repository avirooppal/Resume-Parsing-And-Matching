import re
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_job_description(file_path: str) -> Optional[str]:
    """
    Load job description from a text file.
    
    Args:
        file_path (str): Path to the job description file
        
    Returns:
        Optional[str]: Job description text if successful, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading job description: {str(e)}")
        return None

def extract_required_skills(text: str) -> List[str]:
    """Extract required skills from job description text."""
    skills = []
    
    # Common programming languages and technologies
    tech_keywords = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "spring", "spring boot", "sql", "nosql",
        "mongodb", "postgresql", "mysql", "aws", "azure", "gcp", "docker",
        "kubernetes", "git", "ci/cd", "agile", "scrum", "microservices", "rest", "rest apis"
    ]
    
    text_lower = text.lower()
    logger.debug(f"extract_required_skills - Processing text:\n{text_lower[:500]}...") # Log first 500 chars
    
    # Extract skills from any bullet points in the entire text, being more flexible about whitespace
    bullet_points = re.findall(r'[-•]\s*([^\n]+)', text_lower)
    logger.debug(f"extract_required_skills - Raw bullet points found: {bullet_points}")
    for point in bullet_points:
        for skill in tech_keywords:
            if skill in point:
                if skill.capitalize() not in skills:
                    skills.append(skill.capitalize())
    logger.debug(f"extract_required_skills - Skills after bullet point extraction: {skills}")

    # Extract skills from phrases like "experience in:" or "skills:", followed by a list of skills
    # This pattern now looks for keywords after the colon, potentially across multiple lines
    colon_list_pattern = re.compile(r'(?:experience in:|skills:)\s*\n\s*([\s\S]*?)(?:\n\n|Requirements|\Z)', re.IGNORECASE)
    colon_list_matches = colon_list_pattern.findall(text)
    logger.debug(f"extract_required_skills - Colon list matches: {colon_list_matches}")
    for match_text in colon_list_matches:
        for skill in tech_keywords:
            if re.search(r'\b{}\b'.format(re.escape(skill)), match_text, re.IGNORECASE):
                if skill.capitalize() not in skills:
                    skills.append(skill.capitalize())
    logger.debug(f"extract_required_skills - Skills after colon list extraction: {skills}")

    # Extract skills from the "Requirements" section, looking for skill keywords
    requirements_section_match = re.search(r'requirements:\s*\n([\s\S]*?)(?:\n\n|\Z)', text_lower)
    if requirements_section_match:
        requirements_text = requirements_section_match.group(1)
        logger.debug(f"extract_required_skills - Requirements text: {requirements_text[:200]}...") # Log first 200 chars
        for skill in tech_keywords:
            if re.search(r'\b{}\b'.format(re.escape(skill)), requirements_text):
                if skill.capitalize() not in skills:
                    skills.append(skill.capitalize())
    logger.debug(f"extract_required_skills - Skills after requirements section: {skills}")

    # Extract skills from the entire text if not found in specific sections
    for skill in tech_keywords:
        if re.search(r'\b{}\b'.format(re.escape(skill)), text_lower):
            if skill.capitalize() not in skills:
                skills.append(skill.capitalize())
    logger.debug(f"extract_required_skills - Skills after full text search: {skills}")
    
    return list(set(skills))  # Remove duplicates and ensure unique values

def extract_required_experience(text: str) -> int:
    """Extract required years of experience from job description text."""
    logger.debug(f"extract_required_experience - Processing text:\n{text.lower()[:500]}...") # Log first 500 chars
    # Look for patterns like "X+ years of experience" or "X years experience"
    experience_patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'(\d+)\+?\s*years?\s+with'
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, text.lower())
        logger.debug(f"extract_required_experience - Pattern '{pattern}' match: {match}")
        if match:
            logger.debug(f"extract_required_experience - Extracted years: {int(match.group(1))}")
            return int(match.group(1))
    
    # Look for experience mentioned in requirements section
    requirements_match = re.search(r'requirements:\s*\n[\s\S]*?(\d+)\+?\s*years?\s+of\s+experience', text.lower())
    logger.debug(f"extract_required_experience - Requirements section match: {requirements_match}")
    if requirements_match:
        extracted_years = int(requirements_match.group(1))
        logger.debug(f"extract_required_experience - Extracted years from requirements: {extracted_years}")
        return extracted_years

    logger.debug("extract_required_experience - No experience found, returning 0.")
    return 0  # Default to 0 if not found

def extract_required_education(text: str) -> str:
    """Extract required education level from job description text."""
    education_levels = {
        "bachelor": "Bachelor's Degree",
        "master": "Master's Degree",
        "phd": "PhD",
        "associate": "Associate's Degree",
        "high school": "High School Diploma"
    }
    
    text_lower = text.lower()
    logger.debug(f"extract_required_education - Processing text:\n{text_lower[:500]}...") # Log first 500 chars
    
    for level, degree in education_levels.items():
        match = re.search(rf'{level}[\'s]?(?:\s+degree)?(?:\s+in\s+[\w\s]+)?', text_lower)
        logger.debug(f"extract_required_education - Pattern '{level}' match: {match}")
        if match:
            logger.debug(f"extract_required_education - Extracted education: {degree}")
            return degree
    
    logger.debug("extract_required_education - No education found, returning empty string.")
    return ""  # Return empty string if not found

def parse_job_description(text: str) -> Dict[str, Any]:
    """Parse job description text into structured format.
    
    Args:
        text (str): Job description text
        
    Returns:
        Dict[str, Any]: Parsed job description
    """
    logger.info(f"Parsing job description text:\n{text}")
    
    # Extract title
    title_match = re.search(r'Title:\s*([^\n]+)', text)
    title = title_match.group(1) if title_match else ""
    
    # Extract required skills
    required_skills = extract_required_skills(text)
    logger.info(f"Extracted required skills from JD: {required_skills}")
    
    # Extract required experience
    required_years = extract_required_experience(text)
    logger.info(f"Extracted required years of experience from JD: {required_years}")
    
    # Extract required education
    required_education = extract_required_education(text)
    logger.info(f"Extracted required education from JD: {required_education}")

    parsed_data = {
        "title": title,
        "required_skills": required_skills,
        "required_experience_years": required_years,
        "required_education": required_education,
        "match_text": text  # Store original text for matching
    }

    logger.info(f"Parsed job description data: {parsed_data}")
    return parsed_data 