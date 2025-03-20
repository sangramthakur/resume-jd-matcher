import streamlit as st
import pdfplumber
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Custom stopwords (words to remove)
custom_stopwords = set([
    "experience", "working", "skills", "work", "knowledge", "requirement",
    "good", "strong", "understanding", "hands", "on", "proficient", "year",
    "years", "ability", "required", "develop", "developing", "familiar",
    "excellent", "solid", "background", "expertise", "demonstrated"
])

# Important words that should NOT be removed
keep_words = {"F&O", "CMO", "AI", "ML", "BI", "NLP", "LSTM", "GAN", "IoT", "SQL", "RAG", "LLM"
             "RAG", "LLM", "KPI", "ROI", "OKR", "API", "SDK", "CDN", "SaaS", "PaaS",
                "IaaS", "ETL", "CI/CD", "QA", "UX", "UI", "DevOps", "Docker", "K8s", 
                "AWS", "Azure", "GCP", "HTML", "CSS", "JS", "JSON", "XML", "REST", "SOAP",
                "DBMS", "SQL", "NoSQL", "ETL", "CRM", "ERP", "SEO", "SEM", "PMO", "POC",
                "MVP", "B2B", "B2C", "PMP", "RPA", "AIOps", "QA/QC", "TDD", "BDD", "CI",
                "CD", "Ops", "SRE", "Tech Lead", "API Gateway", "GraphQL", "OAuth", "JWT",
                "Spark", "Hadoop", "Kafka", "ELT", "EDA", "CMS", "DevSecOps", "FP&A", "IoC"
             }

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Extract skills from text while keeping important words
def extract_skills_from_text(text):
    doc = nlp(text)
    words = set([
        token.text for token in doc
        if not token.is_stop and token.is_alpha and 
        token.text.lower() not in custom_stopwords and 
        (len(token.text) > 2 or token.text in keep_words)
    ])
    return words

# Jaccard Similarity Calculation
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

# Cosine Similarity Calculation
def cosine_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors)[0, 1]

# Evaluate Skill Extraction (Precision, Recall, F1-score)
def evaluate_extraction(predicted, actual):
    true_positive = len(set(predicted) & set(actual))
    false_positive = len(set(predicted) - set(actual))
    false_negative = len(set(actual) - set(predicted))

    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)

    return {"Precision": round(precision, 2), "Recall": round(recall, 2), "F1-score": round(f1, 2)}

# Streamlit App
st.title("📄 Resume vs JD Skill Matcher")
st.write("Upload your Resume & Job Description PDFs or copy-paste the JD text.")

# Resume Upload/Input
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume_text_input = st.text_area("Or, Copy-Paste Resume Here", height=200)

# JD Upload/Input
jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
jd_text_input = st.text_area("Or, Copy-Paste JD Here", height=200)

if (resume_file or resume_text_input) and (jd_file or jd_text_input):
    # Extract Resume & JD Text
    resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_text_input.strip()
    jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_text_input.strip()

    # Extract Named Entities from Resume
    resume_entities = extract_entities(resume_text)

    # Extract Skills from Resume & JD
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    # Evaluate Skill Matching
    skill_evaluation = evaluate_extraction(resume_skills, jd_skills)

    # Compute Similarity Scores
    jaccard_score = jaccard_similarity(resume_text, jd_text)
    cosine_score = cosine_similarity_score(resume_text, jd_text)

    # Skills Comparison
    common_skills = sorted(resume_skills & jd_skills)
    resume_unique = sorted(resume_skills - jd_skills)
    jd_unique = sorted(jd_skills - resume_skills)

    # Display Extracted Entities
    st.subheader("📌 Extracted Resume Information")
    st.json(resume_entities)

    # Display Skill Extraction Evaluation
    st.subheader("📊 Skill Extraction Evaluation")
    st.json(skill_evaluation)

    # Display Similarity Scores
    st.subheader("📏 Resume vs JD Similarity")
    st.write(f"🔹 **Jaccard Similarity:** `{round(jaccard_score, 2)}`")
    st.write(f"🔹 **Cosine Similarity:** `{round(cosine_score, 2)}`")

    # Create DataFrame for Skills Comparison
    skills_df = pd.DataFrame({
        "Common Skills": common_skills + [""] * (max(len(resume_unique), len(jd_unique)) - len(common_skills)),
        "Resume Only Skills": resume_unique + [""] * (max(len(common_skills), len(jd_unique)) - len(resume_unique)),
        "JD Only Skills": jd_unique + [""] * (max(len(common_skills), len(resume_unique)) - len(jd_unique)),
    })

    # Display Skills Table
    st.subheader("📝 Skills Comparison Table")
    st.dataframe(skills_df)

    # Allow CSV Download
    csv = skills_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Skills Comparison as CSV", csv, "skills_comparison.csv", "text/csv")
