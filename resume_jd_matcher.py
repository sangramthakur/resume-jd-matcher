import streamlit as st
import pandas as pd
import spacy
import re
import json
import tika
from tika import parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import docx
import pytesseract
from PIL import Image

# Initialize Apache Tika
tika.initVM()

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Custom stopwords
custom_stopwords = set([
    "experience", "working", "skills", "work", "knowledge", "requirement",
    "good", "strong", "understanding", "hands", "on", "proficient", "year",
    "years", "ability", "required", "develop", "developing", "familiar",
    "excellent", "solid", "background", "expertise", "demonstrated"
])

# Important words that should NOT be removed
keep_words = {
    "F&O", "CMO", "AI", "ML", "BI", "NLP", "LSTM", "GAN", "IoT", "SQL", "RAG", "LLM",
    "KPI", "ROI", "OKR", "API", "SDK", "CDN", "SaaS", "PaaS", "IaaS", "ETL", "CI/CD",
    "Docker", "AWS", "Azure", "GCP", "REST", "DBMS", "NoSQL", "CRM", "ERP", PMP, PgMP, CISSP, CISA, CISM
}

# Function to extract text using Apache Tika
def extract_text(file):
    parsed = parser.from_buffer(file)
    return parsed['content'].strip() if parsed and 'content' in parsed and parsed['content'] else ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from CSV
def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return " ".join(df.astype(str).values.flatten())

# Function to extract text from JSON
def extract_text_from_json(json_file):
    try:
        data = json.load(json_file)
        return " ".join(str(value) for value in data.values()) if isinstance(data, dict) else str(data)
    except Exception:
        return ""

# OCR Function for Scanned PDFs & Images
def ocr_extract_text(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Function to process file based on type
def process_file(file):
    file_type = file.name.split('.')[-1].lower()

    if file_type in ["pdf", "docx", "rtf", "txt"]:
        return extract_text(file)

    elif file_type in ["csv"]:
        return extract_text_from_csv(file)

    elif file_type in ["json"]:
        return extract_text_from_json(file)

    elif file_type in ["jpg", "jpeg", "png", "tiff"]:
        return ocr_extract_text(file)

    else:
        return ""

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    words = set([
        token.text for token in doc
        if not token.is_stop and token.is_alpha and
        token.text.lower() not in custom_stopwords and
        (len(token.text) > 2 or token.text in keep_words)
    ])
    return words

# Similarity Calculations
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def cosine_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors)[0, 1]

# Streamlit UI
st.title("📄 Resume vs JD Matcher with OCR & Tika")
st.write("Upload Resume, Job Description, and Mandatory Skills in multiple formats!")

# Resume Upload
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "rtf", "txt", "csv", "json", "jpg", "jpeg", "png", "tiff"])
resume_text_input = st.text_area("Or, Copy-Paste Resume Here", height=200)

# JD Upload
jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "rtf", "txt", "csv", "json", "jpg", "jpeg", "png", "tiff"])
jd_text_input = st.text_area("Or, Copy-Paste JD Here", height=200)

# Mandatory Skills Upload
mandatory_skills_file = st.file_uploader("Upload Mandatory Skills", type=["pdf", "docx", "rtf", "txt", "csv", "json", "jpg", "jpeg", "png", "tiff"])
mandatory_skills_text_input = st.text_area("Or, Copy-Paste Mandatory Skills Here", height=100)

# Process Inputs
if (resume_file or resume_text_input) and (jd_file or jd_text_input):
    # Extract Resume & JD Text
    resume_text = process_file(resume_file) if resume_file else resume_text_input.strip()
    jd_text = process_file(jd_file) if jd_file else jd_text_input.strip()

    # Extract Skills from Resume & JD
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # Compute Similarity Scores
    jaccard_score = jaccard_similarity(resume_text, jd_text)
    cosine_score = cosine_similarity_score(resume_text, jd_text)

    # Mandatory Skills Extraction
    mandatory_skills_text = process_file(mandatory_skills_file) if mandatory_skills_file else mandatory_skills_text_input
    mandatory_skills = extract_skills(mandatory_skills_text)

    # Compare Skills
    common_skills = sorted(resume_skills & jd_skills)
    resume_unique = sorted(resume_skills - jd_skills)
    jd_unique = sorted(jd_skills - resume_skills)

    # Compare Mandatory Skills
    mandatory_match = sorted(resume_skills & mandatory_skills)
    missing_mandatory = sorted(mandatory_skills - resume_skills)

    # Display Results
    st.subheader("📏 Resume vs JD Similarity")
    st.write(f"🔹 **Jaccard Similarity:** `{round(jaccard_score, 2)}`")
    st.write(f"🔹 **Cosine Similarity:** `{round(cosine_score, 2)}`")

    # Skills Comparison Table
    skills_df = pd.DataFrame({
        "Common Skills": common_skills + [""] * (max(len(resume_unique), len(jd_unique)) - len(common_skills)),
        "Resume Only Skills": resume_unique + [""] * (max(len(common_skills), len(jd_unique)) - len(resume_unique)),
        "JD Only Skills": jd_unique + [""] * (max(len(common_skills), len(resume_unique)) - len(jd_unique)),
    })
    st.subheader("📝 Skills Comparison Table")
    st.dataframe(skills_df)

    # Mandatory Skills Matching
    st.subheader("✅ Mandatory Skills Match")
    st.write(f"✅ **Matched Mandatory Skills:** {', '.join(mandatory_match) if mandatory_match else 'None'}")
    st.write(f"❌ **Missing Mandatory Skills:** {', '.join(missing_mandatory) if missing_mandatory else 'None'}")

    # Download CSV
    csv = skills_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Skills Comparison as CSV", csv, "skills_comparison.csv", "text/csv")
