from typing import Dict, Any, List
import spacy
from transformers import pipeline

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_entities_with_llm(text: str) -> Dict[str, Any]:
    """
    Extract entities from text using LLM-based NER.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, Any]: Extracted entities
    """
    # Get NER predictions
    ner_results = ner_pipeline(text)
    
    # Process entities
    entities = {
        "name": "",
        "email": "",
        "phone": "",
        "job_title": "",
        "summary": "",
        "skills": [],
        "education": [],
        "work": [],
        "location_parts": []  # Initialize as a list to collect parts
    }
    
    # Process NER results
    current_entity = ""
    current_type = ""
    
    for result in ner_results:
        word = result["word"]
        entity_type = result["entity"]
        
        if entity_type.startswith("B-"):  # Beginning of entity
            if current_entity:
                add_entity_to_dict(entities, current_entity, current_type)
            current_entity = word
            current_type = entity_type[2:]
        elif entity_type.startswith("I-"):  # Inside entity
            current_entity += " " + word
        else:  # Single token entity
            if current_entity:
                add_entity_to_dict(entities, current_entity, current_type)
            add_entity_to_dict(entities, word, entity_type)
            current_entity = ""
            current_type = ""
    
    # Add last entity if exists
    if current_entity:
        add_entity_to_dict(entities, current_entity, current_type)

    # Join location parts into a single string
    entities["location"] = ", ".join(entities["location_parts"])
    del entities["location_parts"] # Remove the temporary list

    # Extract summary using spaCy
    doc = nlp(text)
    summary = ""
    for sent in doc.sents:
        if len(summary) < 200:  # Limit summary length
            summary += sent.text + " "
    entities["summary"] = summary.strip()
    
    return entities

def add_entity_to_dict(entities: Dict[str, Any], entity: str, entity_type: str):
    """Add extracted entity to the entities dictionary."""
    if entity_type == "PER":
        if not entities["name"]:
            entities["name"] = entity
    elif entity_type == "ORG":
        if not entities["job_title"]:
            entities["job_title"] = entity
    elif entity_type == "MISC":
        entities["skills"].append({
            "name": entity,
            "level": "Intermediate"
        })
    elif entity_type == "LOC":
        entities["location_parts"].append(entity) 