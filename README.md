# 📄 Intelligent Resume/CV Keyword Matcher & Parser

An interactive Natural Language Processing (NLP) web application built with **Python**, **Streamlit**, **NLTK**, and **Scikit-learn**. This tool parses PDF resumes, analyzes text content against target Job Descriptions (JD), calculates semantic match scores, and identifies missing key skills to help job seekers optimize their resumes for Applicant Tracking Systems (ATS).

🚀 **Live Application:** [Launch Resume Matcher App](https://resume-jd-matcher-zwupsm5cjxxivjjig5nxxj.streamlit.app/)

---

## ✨ Features

- **PDF Resume Parsing**: Extracts raw text cleanly from uploaded PDF resumes using `pdfplumber`.
- **Text Preprocessing & Tokenization**: Cleans, normalizes, tokenizes, removes stopwords, and performs lemmatization using NLTK WordNet.
- **Semantic Vectorization & Match Scoring**: Converts text into TF-IDF vector representations and computes Cosine Similarity match percentages.
- **Missing Keyword Gap Analysis**: Highlights specific skills and keywords present in the Job Description but absent from the candidate's resume.
- **Interactive UI**: Built with Streamlit for a fast, responsive, and seamless user experience.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: Streamlit
- **NLP & Text Processing**: NLTK (Tokenization, Stopwords Removal, WordNet Lemmatizer)
- **Machine Learning & Vectorization**: Scikit-learn (TfidfVectorizer, cosine_similarity)
- **PDF Extraction**: `pdfplumber`
- **Language**: Python 3.10+

---

## 📁 Repository Structure

```text
resume-jd-matcher/
- app.py (Main Streamlit web application layout & UI logic)
- nlp_engine.py (Text extraction, preprocessing, TF-IDF vectorization & scoring)
- requirements.txt (Project dependencies)
- .gitignore (Git ignore rules)
