from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Optional
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
PORT = int(os.getenv("PORT", 8000))

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_processor import extract_text_from_pdf
from utils.section_entity_extraction import extract_section_entities
from utils.job_description_parser import parse_job_description
from utils.match_scoring import calculate_match_score
from utils.models import load_models
from utils.file_handler import load_job_description

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != '*' else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Load models once at startup
models = None

@app.on_event("startup")
async def startup_event():
    global models
    models = load_models()

from typing import List

import asyncio

MAX_FILE_SIZE_MB = 10
ALLOWED_RESUME_EXT = {'.pdf'}
ALLOWED_JD_EXT = {'.txt'}
TIMEOUT_SECONDS = 60

@app.post("/api/match")
async def match_resume(
    resumes: List[UploadFile] = File(...),
    job_description: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = None
):
    results_list = []
    jd_path = None
    try:
        async def process():
            # Validate resumes
            for resume in resumes:
                ext = os.path.splitext(resume.filename)[1].lower()
                if ext not in ALLOWED_RESUME_EXT:
                    raise HTTPException(status_code=400, detail=f"Invalid resume file type: {resume.filename}. Only PDF allowed.")
                resume.file.seek(0, 2)
                size_mb = resume.file.tell() / (1024 * 1024)
                resume.file.seek(0)
                if size_mb > MAX_FILE_SIZE_MB:
                    raise HTTPException(status_code=400, detail=f"Resume file too large: {resume.filename}. Max {MAX_FILE_SIZE_MB}MB allowed.")

            # Validate job description file
            if job_description:
                ext = os.path.splitext(job_description.filename)[1].lower()
                if ext not in ALLOWED_JD_EXT:
                    raise HTTPException(status_code=400, detail=f"Invalid job description file type: {job_description.filename}. Only TXT allowed.")
                job_description.file.seek(0, 2)
                size_mb = job_description.file.tell() / (1024 * 1024)
                job_description.file.seek(0)
                if size_mb > MAX_FILE_SIZE_MB:
                    raise HTTPException(status_code=400, detail=f"Job description file too large: {job_description.filename}. Max {MAX_FILE_SIZE_MB}MB allowed.")

            # Process job description
            if job_description:
                nonlocal jd_path
                jd_path = os.path.join(UPLOAD_DIR, job_description.filename)
                with open(jd_path, "wb") as buffer:
                    shutil.copyfileobj(job_description.file, buffer)
                jd_text_val = load_job_description(jd_path)
            elif not jd_text:
                raise HTTPException(status_code=400, detail="Either job description file or text must be provided")
            else:
                jd_text_val = jd_text

            # Parse job description once
            parsed_jd = parse_job_description(jd_text_val)

            for resume in resumes:
                try:
                    # Save uploaded resume
                    resume_path = os.path.join(UPLOAD_DIR, resume.filename)
                    with open(resume_path, "wb") as buffer:
                        shutil.copyfileobj(resume.file, buffer)

                    # Extract text from resume
                    try:
                        resume_text = extract_text_from_pdf(resume_path)
                    except Exception as pdf_err:
                        raise HTTPException(status_code=400, detail=f"PDF extraction failed for {resume.filename}: {pdf_err}")
                    if not resume_text:
                        raise HTTPException(status_code=400, detail=f"Could not extract text from resume: {resume.filename}")

                    # Parse resume
                    parsed_resume = extract_section_entities(resume_text, models["ner_model"])

                    # Calculate match scores
                    match_results = calculate_match_score(parsed_resume, parsed_jd, models["embedding_model"])

                    # Flatten match_results.details into match_results for frontend compatibility
                    if "details" in match_results:
                        match_results.update(match_results.pop("details"))
                    # Ensure resume fields are always present
                    resume_response = dict(parsed_resume)
                    for field in ["name", "email", "phone", "location"]:
                        if field not in resume_response:
                            resume_response[field] = ""
                    results = {
                        "resume": resume_response,
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
                    resume_name = os.path.splitext(os.path.basename(resume_path))[0]
                    jd_name = "job_description" if not job_description else os.path.splitext(os.path.basename(jd_path))[0]
                    output_file = os.path.join(OUTPUT_DIR, f"{resume_name}_vs_{jd_name}_match.json")

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)

                    results_list.append(results)

                except HTTPException as he:
                    results_list.append({"error": he.detail, "filename": resume.filename})
                except Exception as e:
                    results_list.append({"error": str(e), "filename": resume.filename})
                finally:
                    # Cleanup uploaded resume file
                    if os.path.exists(resume_path):
                        os.remove(resume_path)

            # Cleanup job description file
            if job_description and jd_path and os.path.exists(jd_path):
                os.remove(jd_path)

            return {"results": results_list}

        return await asyncio.wait_for(process(), timeout=TIMEOUT_SECONDS)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Request timed out after {TIMEOUT_SECONDS} seconds.")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(feedback: dict):
    feedback_text = feedback.get("feedback_text")
    if not feedback_text:
        raise HTTPException(status_code=400, detail="Feedback text is required")
    
    try:
        feedback_file_path = os.path.join(OUTPUT_DIR, "feedback.txt")
        with open(feedback_file_path, "a", encoding="utf-8") as f:
            f.write(feedback_text + "\n---\n")
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="debug") 