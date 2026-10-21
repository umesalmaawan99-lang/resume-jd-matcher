import streamlit as st
from nlp_engine import extract_text_from_pdf, calculate_match_score, extract_missing_keywords

# Page Configuration
st.set_page_config(
    page_title="Intelligent Resume & JD Matcher",
    page_icon="📄",
    layout="wide"
)

# App Header
st.title("📄 Intelligent Resume / CV Keyword Matcher & Parser")
st.markdown("Upload your resume, paste a target job description, and discover your semantic match score along with missing skills!")

# Sidebar for tips
with st.sidebar:
    st.header("💡 How to Use")
    st.markdown("""
    1. **Upload Resume:** Select a PDF version of your CV.
    2. **Paste Job Description:** Copy and paste the full job posting into the text area.
    3. **Analyze Match:** Click the button to run TF-IDF and Cosine Similarity analysis.
    4. **Optimize:** Review your match percentage and target the missing keywords!
    """)
    st.markdown("---")
    st.markdown("🚀 Built with Python, Scikit-Learn & Streamlit")

# Main Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Choose your resume PDF file", type=["pdf"])
    
    resume_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            st.success("Resume successfully uploaded and parsed!")
            with st.expander("🔍 View Extracted Resume Text"):
                st.text_area("Parsed Text", resume_text, height=200, disabled=True)

with col2:
    st.subheader("2. Paste Job Description")
    jd_text = st.text_area("Paste target Job Description here...", height=250)

st.markdown("---")

# Analysis Section
if st.button("🚀 Analyze Resume & Job Match", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ Please upload a resume PDF first.")
    elif not jd_text.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
    # Calculate Results
        with st.spinner("Running NLP similarity analysis..."):
            match_score = calculate_match_score(resume_text, jd_text)
            missing_keywords = extract_missing_keywords(resume_text, jd_text, top_n=10)
        
        # Display Results Dashboard
        st.header("📊 Match Analysis Results")
        
        # Score Display Metric Card
        col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
        with col_m2:
            st.metric(label="Overall Semantic Match Score", value=f"{match_score}%")
            st.progress(match_score / 100.0)
            
        st.markdown("### 🔍 Key Insights & Recommendations")
        
        if match_score >= 75:
            st.success("🌟 **Strong Match!** Your resume aligns very well with this job description.")
        elif match_score >= 50:
            st.info("👍 **Moderate Match.** Good alignment, but tailoring your skills could boost your chances.")
        else:
            st.warning("⚠️ **Low Match.** Consider incorporating more core keywords from the job description.")
            
        # Missing Keywords Section
        st.subheader("🔑 Top Missing Keywords / Skills to Add")
        if missing_keywords:
            st.markdown("Your resume appears to be missing these key terms found in the job description:")
            # Display chips/badges style using markdown
            keyword_badges = " ".join([f"`{kw}`" for kw in missing_keywords])
            st.markdown(keyword_badges)
        else:
            st.success("🎉 Great job! No major high-weight keywords from the job description are missing.")