

# 🧾 Resume Match Pipeline

A robust, end-to-end pipeline for parsing resumes and job descriptions, extracting structured information, and intelligently matching candidates to jobs using both rule-based and semantic (AI/ML) techniques. Includes a web frontend and a FastAPI backend, fully Dockerized for easy deployment.

---

## 🚀 How It Works: Step-by-Step

### 1. **Resume & Job Description Upload**
- Users upload one or more resumes (PDF) and a job description (TXT or pasted text) via the web interface.

### 2. **Parsing & Extraction**
#### 📄 Resume Parsing
- PDF text extraction.
- Section detection (summary, skills, work, education, etc.) using regex and heuristics.
- Named Entity Recognition (NER) using **Flair** for names, locations, etc.
- Contact info extraction using regex + NER (with fallbacks for edge cases).

#### 📑 Job Description Parsing
- Extract required skills, experience, and education using regex and keyword matching.

### 3. **Matching Logic**
#### ✅ Skill Matching
- Exact skill matches.
- Semantic skill matches using **SentenceTransformer** (BERT-based embeddings).

#### 📊 Experience & Education
- Calculate years of experience from work history and compare to job requirements.
- Match education via string and keyword equivalence.

#### 🧮 Scoring
- Weighted scoring for:
  - Skills (exact + semantic)
  - Experience
  - Education
- Detailed breakdown:
  - Matched / Missing / Semantically matched skills
  - Education and experience match status

### 4. **Frontend Display**
- Resume summaries with scoring.
- Select any resume from a dropdown to view detailed match and parsing results.

---

## 🧠 Model Stack

| Task                     | Library/Model                                                                 |
|--------------------------|-------------------------------------------------------------------------------|
| Named Entity Recognition | [Flair](https://github.com/flairNLP/flair)                                   |
| Semantic Matching        | [SentenceTransformers](https://www.sbert.net/)                                |
| Parsing & Extraction     | Regex + heuristics                                                            |
| Text Processing          | Gensim, langdetect, etc.                                                      |

---

## 🐳 Running the Project from Scratch

### 1. **Clone the Repository**
```bash
git clone https://github.com/avirooppal/resume-match-pipeline.git
cd resume-match-pipeline
````

### 2. **(Recommended) Use Docker**

Build and run the app (ensure Docker is installed):

```bash
docker build -t resume-match-app .
docker run -p 8000:8000 resume-match-app
```

* Access the app at: [http://localhost:8000](http://localhost:8000)
* Frontend is served from `/static`:
  [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

### 3. **(Alternative) Run Locally with Python**

Install Python 3.8 and create a virtual environment:

```bash
python3.8 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Download required models:

```bash
python models.py
```

Start the backend:

```bash
cd web
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open frontend:

* File: `web/static/index.html`
* Or access via browser: `http://localhost:8000/static/index.html`

### 4. **Environment Variables**

* Configure a `.env` file for:

  * Frontend/backend URLs
  * Secrets or model paths (if applicable)

---

## 📝 Notes

* **Multiple Resumes**: Upload and compare multiple resumes; switch between them in the UI.
* **Error Handling**: File type validation, size checks, PDF parsing errors, and timeouts handled robustly.
* **Extensible**: Easy to extend:

  * Add new ML models or scoring logic
  * Customize UI or add filtering/sorting
  * Integrate with ATS or HR tools
