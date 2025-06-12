import re
import string
from config.config import config
from utils.file_handler import load_skills_ontology, load_job_title_mapping
from symspellpy import SymSpell

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_stopwords(text):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def correct_typo(text):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(config['skills_ontology_path'], term_index=0, count_index=1)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = correct_typo(text)
    return text

def extract_sections(text):
    section_headers = load_skills_ontology()['section_headers']
    sections = {}
    current_section = None
    lines = text.split('\n')
    for line in lines:
        for header in section_headers:
            if header in line:
                current_section = header
                sections[current_section] = []
                break
        if current_section:
            sections[current_section].append(line)
    return sections