import re
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')

download_nltk_resources()

def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file object using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    return text

def preprocess_text(text):
    """Cleans text: lowercase, remove special characters/numbers, tokenization, stopword removal, and lemmatization."""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers, keeping only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    return " ".join(cleaned_tokens)

def calculate_match_score(resume_text, jd_text):
    """Calculates cosine similarity percentage score between resume and job description."""
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)
    
    if not processed_resume or not processed_jd:
        return 0.0
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
    
    # Compute Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = float(similarity[0][0]) * 100
    
    return round(score, 2)

def extract_missing_keywords(resume_text, jd_text, top_n=10):
    """Identifies top keywords/skills in the Job Description missing from the resume."""
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)
    
    if not processed_resume or not processed_jd:
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_jd, processed_resume])
        feature_names = vectorizer.get_feature_names_out()
        
        jd_vector = tfidf_matrix.toarray()[0]
        resume_vector = tfidf_matrix.toarray()[1]
        
        # Find terms with high weight in JD but zero/low weight in Resume
        missing_indices = []
        for idx, (jd_w, res_w) in enumerate(zip(jd_vector, resume_vector)):
            if jd_w > 0 and res_w == 0:
                missing_indices.append((idx, jd_w))
                
        # Sort by highest TF-IDF score in JD
        missing_indices.sort(key=lambda x: x[1], reverse=True)
        
        missing_keywords = [feature_names[idx] for idx, _ in missing_indices[:top_n]]
        return missing_keywords
    except Exception:
        return []