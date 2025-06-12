import json
import os
from config.config import config
from typing import Dict, Any
from utils.pdf_processor import extract_text_from_pdf

def read_pdf(file_path):
    import pdfminer.high_level
    with open(file_path, 'rb') as file:
        text = pdfminer.high_level.extract_text(file)
    return text

def read_docx(file_path):
    import docx
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def read_rtf(file_path):
    import striprtf
    with open(file_path, 'r') as file:
        rtf = file.read()
    text, _ = striprtf.rtf_to_text(rtf)
    return text

def read_file(file_path: str) -> str:
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""

def load_resume(resume_path: str) -> str:
    """Load resume text from PDF or text file."""
    if resume_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(resume_path)
    else:
        return read_file(resume_path)

def load_job_description(jd_path: str) -> str:
    """Load job description text from file."""
    return read_file(jd_path)

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

def load_feedback_data():
    with open(config['feedback_data_path'], 'r') as file:
        return json.load(file)

def save_feedback_data(feedback_data):
    with open(config['feedback_data_path'], 'w') as file:
        json.dump(feedback_data, file, indent=4)

def save_match_results(results: Dict[str, Any]) -> None:
    """Save match results to a JSON file."""
    output_dir = os.path.join(config["base_dir"], "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "match_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)