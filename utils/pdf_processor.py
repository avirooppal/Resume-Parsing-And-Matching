import fitz  # PyMuPDF
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text if successful, None otherwise
    """
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            logger.info(f"Extracted text from page {page_num + 1}:\n{page_text}\n")
        
        # Close the document
        doc.close()
        
        extracted_text = text.strip()
        logger.info(f"Total extracted text length: {len(extracted_text)} characters")
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None 