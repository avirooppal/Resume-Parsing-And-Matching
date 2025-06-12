import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_skills(text: str) -> List[Dict[str, str]]:
    """Extract skills using keyword matching, regex patterns, and spaCy."""
    skills = []
    
    # Common programming languages and technologies
    tech_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c#", "c++", "ruby", "php", "go", "rust",
        # Web Technologies
        "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring",
        # Databases
        "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis", "cassandra",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "ci/cd", "terraform",
        # Frameworks & Tools
        "spring boot", "hibernate", "junit", "maven", "gradle", "npm", "yarn", "webpack",
        # Methodologies
        "agile", "scrum", "devops", "tdd", "bdd", "microservices", "rest", "graphql",
        # Other
        "linux", "unix", "shell scripting", "bash", "powershell"
    ]
    
    # Normalize text
    text_lower = text.lower()
    
    # Regex patterns for skill sections
    skill_section_pattern = re.compile(
        r'(skills|technologies|expertise|competencies|technical skills)[^\S\r\n]*[:\-—]?\s*((?:.|\n)*?)(?=\n\n|\Z)',
        re.IGNORECASE
    )
    
    # Find skill sections
    for section_match in skill_section_pattern.finditer(text_lower):
        section_content = section_match.group(2)
        
        # Split by commas, hyphens, or newlines
        skill_candidates = re.split(r'[,\n]|[-—–]\s*', section_content)
        
        for skill_candidate in skill_candidates:
            skill_candidate = re.sub(r'[^\w\s]', '', skill_candidate.strip())
            if not skill_candidate:
                continue
                
            # Exact match
            if skill_candidate.lower() in tech_keywords:
                skills.append({
                    "name": skill_candidate.capitalize(),
                    "level": "Intermediate"
                })
            # Fuzzy match for multi-word skills
            else:
                for keyword in tech_keywords:
                    if keyword in skill_candidate.lower():
                        skills.append({
                            "name": keyword.capitalize(),
                            "level": "Intermediate"
                        })
    
    # Remove duplicates
    return list({s["name"]: s for s in skills}.values())

def extract_section_entities(text: str) -> Dict[str, Any]:
    """Extract entities from resume text using section-based approach with improved header detection."""
    # Normalize text
    text = text.replace('\r', '\n')
    
    # Define section headers with regex patterns
    section_patterns = {
        "work": re.compile(r'(experience|work|employment|professional experience)', re.IGNORECASE),
        "education": re.compile(r'(education|academic|degree|certifications)', re.IGNORECASE),
        "skills": re.compile(r'(skills|technologies|expertise|competencies|technical skills)', re.IGNORECASE),
        "contact": re.compile(r'(contact|personal info|about me)', re.IGNORECASE)
    }
    
    # Split text into sections based on headers
    sections = []
    current_pos = 0
    header_pattern = re.compile(r'(\n|^)\s*(' + '|'.join(section_patterns.keys()) + r')\s*(?=\n|$)', re.IGNORECASE)
    
    for match in header_pattern.finditer(text):
        if current_pos < match.start():
            sections.append(text[current_pos:match.start()].strip())
        current_pos = match.end()
    
    if current_pos < len(text):
        sections.append(text[current_pos:].strip())
    
    # Initialize entities dictionary
    entities = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text),
        "work": [],
        "education": [],
        "skills": []
    }
    
    # Process each section
    for section in sections:
        if not section:
            continue
            
        # Identify section type
        section_type = None
        for key, pattern in section_patterns.items():
            if pattern.search(section):
                section_type = key
                break
                
        if section_type == "work":
            work_experiences = extract_work_experience(section)
            entities["work"].extend(work_experiences)
        elif section_type == "education":
            education_entries = extract_education(section)
            entities["education"].extend(education_entries)
        elif section_type == "skills":
            skills = extract_skills(section)
            entities["skills"].extend(skills)
    
    return entities

def extract_education(text: str) -> List[Dict[str, Any]]:
    """Extract education entries with improved parsing."""
    education_entries = []
    current_education = {}
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for degree keywords
        degree_match = re.search(r'(bachelor|master|phd|b\.s\.|m\.s\.|b\.a\.|m\.a\.)', line.lower())
        if degree_match:
            if current_education:
                education_entries.append(current_education)
            current_education = {
                "institution": "",
                "area": line,
                "studyType": "",
                "startDate": "",
                "endDate": ""
            }
        # Look for dates
        elif re.search(r'\d{4}', line):
            dates = re.findall(r'\d{4}', line)
            if len(dates) >= 2:
                current_education["startDate"] = dates[0]
                current_education["endDate"] = dates[1]
            elif len(dates) == 1:
                current_education["startDate"] = dates[0]
                current_education["endDate"] = "Present"
        # Look for institutions
        elif re.search(r'(university|college|institute|school)', line.lower()):
            current_education["institution"] = line
    
    if current_education:
        education_entries.append(current_education)
    
    return education_entries

def extract_work_experience(text: str) -> List[Dict[str, Any]]:
    """Extract work experience entries with improved parsing."""
    work_experiences = []
    current_experience = {}
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for company names
        if re.search(r'(inc\.?|llc|ltd\.?|corp\.?|company|corporation|technologies|solutions|systems)', line.lower()):
            if current_experience:
                work_experiences.append(current_experience)
            current_experience = {
                "company": line,
                "position": "",
                "startDate": "",
                "endDate": "",
                "summary": ""
            }
        # Look for dates
        elif re.search(r'\d{4}', line):
            dates = re.findall(r'\d{4}', line)
            if len(dates) >= 2:
                current_experience["startDate"] = dates[0]
                current_experience["endDate"] = dates[1]
            elif len(dates) == 1:
                current_experience["startDate"] = dates[0]
                current_experience["endDate"] = "Present"
        # Look for job titles
        elif re.search(r'(engineer|developer|manager|director|lead|architect|consultant|analyst|specialist)', line.lower()):
            if not current_experience.get("position"):
                current_experience["position"] = line
        # Add to summary
        else:
            if current_experience.get("summary"):
                current_experience["summary"] += " " + line
            else:
                current_experience["summary"] = line
    
    if current_experience:
        work_experiences.append(current_experience)
    
    return work_experiences

def calculate_skill_match(resume_skills: List[Dict[str, str]], jd_skills: List[str], model: SentenceTransformer) -> Tuple[List[str], List[str], List[str]]:
    """Calculate skill matches between resume and job description with improved handling."""
    matched_skills = []
    missing_skills = []
    semantically_matched_skills = []
    
    # Convert resume skills to set for faster lookup
    resume_skill_set = {skill["name"].lower() for skill in resume_skills}
    
    # Check for exact matches
    for skill in jd_skills:
        skill_lower = skill.lower()
        if skill_lower in resume_skill_set:
            matched_skills.append(skill)
        else:
            # Check for semantic matches
            for resume_skill in resume_skills:
                similarity = calculate_semantic_similarity(skill, resume_skill["name"], model)
                if similarity > 0.7:
                    semantically_matched_skills.append(skill)
                    matched_skills.append(skill)
                    break
            else:
                missing_skills.append(skill)
    
    return matched_skills, missing_skills, semantically_matched_skills

def calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate semantic similarity between two texts using sentence transformers."""
    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return float(similarity) 