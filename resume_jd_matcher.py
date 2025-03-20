import streamlit as st
import pdfplumber
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Custom stop words
custom_stopwords = set([
    "experience", "working", "skills", "work", "knowledge", "requirement",
    "good", "strong", "understanding", "hands", "on", "proficient", "year",
    "years", "ability", "required", "develop", "developing", "familiar",
    "excellent", "solid", "background", "expertise", "demonstrated"
])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Function to extract named entities (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Function to extract skills with stopword removal
def extract_skills_from_text(text):
    doc = nlp(text)
    words = set([
        token.text.lower() for token in doc
        if not token.is_stop and token.is_alpha and token.text.lower() not in custom_stopwords and len(token.text) > 2
    ])
    return words

# Function to calculate Jaccard Similarity
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

# Function to calculate Cosine Similarity
def cosine_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors)[0, 1]

# Function to evaluate skill extraction
def evaluate_extraction(predicted, actual):
    true_positive = len(set(predicted) & set(actual))
    false_positive = len(set(predicted) - set(actual))
    false_negative = len(set(actual) - set(predicted))

    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)

    return {"Precision": round(precision, 2), "Recall": round(recall, 2), "F1-score": round(f1, 2)}

# Streamlit UI
st.title("📄 Resume vs JD Skill Matcher")
st.write("Upload your Resume & Job Description PDFs or copy-paste the JD text.")

# File Upload for Resume OR Copy-Paste Resume
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume_text_input = st.text_area("Or, Copy-Paste Resume Here", height=200)

# File Upload for JD OR Copy-Paste JD
jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
jd_text_input = st.text_area("Or, Copy-Paste JD Here", height=200)

if (resume_file or resume_text_input) and (jd_file or jd_text_input):
    # Extract text from Resume PDF or use pasted text
    resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_text_input.strip()

    # Extract text from JD PDF or use pasted text
    jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_text_input.strip()

    # Extract entities from Resume
    resume_entities = extract_entities(resume_text)

    # Extract skills from Resume & JD
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)  # JD skills as ground truth

    # Evaluate skill extraction using JD skills as actual skills
    skill_evaluation = evaluate_extraction(resume_skills, jd_skills)

    # Calculate Similarity Scores
    jaccard_score = jaccard_similarity(resume_text, jd_text)
    cosine_score = cosine_similarity_score(resume_text, jd_text)

    # Identify common and unique skills
    common_skills = sorted(resume_skills & jd_skills)
    resume_unique = sorted(resume_skills - jd_skills)
    jd_unique = sorted(jd_skills - resume_skills)

    # Create a Pandas DataFrame for tabular display
    max_len = max(len(common_skills), len(resume_unique), len(jd_unique))
    data = {
        "Common Skills": common_skills + [""] * (max_len - len(common_skills)),
        "Resume-Only Skills": resume_unique + [""] * (max_len - len(resume_unique)),
        "JD-Only Skills": jd_unique + [""] * (max_len - len(jd_unique)),
    }
    skills_df = pd.DataFrame(data)

    # Display Extracted Information
    st.subheader("📌 Extracted Resume Information")
    st.json(resume_entities)

    st.subheader("📊 Skill Extraction Evaluation")
    st.json(skill_evaluation)

    st.subheader("📏 Resume vs JD Similarity")
    st.write(f"🔹 **Jaccard Similarity:** `{round(jaccard_score, 2)}`")
    st.write(f"🔹 **Cosine Similarity:** `{round(cosine_score, 2)}`")

    st.subheader("📝 Skills Comparison Table")
    st.dataframe(skills_df)

    # Option to Download CSV
    csv = skills_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Skills Comparison as CSV", csv, "skills_comparison.csv", "text/csv")

# Run with: `streamlit run script.py`
# Open browser: http://localhost:8501
