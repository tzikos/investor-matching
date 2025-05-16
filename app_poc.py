import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pptx import Presentation
import io

st.set_page_config(page_title="Startup-Investor Matcher", page_icon="🚀")

st.title("🚀 Startup Pitch to Investor Matcher")
st.markdown("**Connect with the right investors for your startup idea.**\nUpload your pitch or write it below to find your top 3 investor matches based on LinkedIn data. 💼✨")

# Upload investors.xlsx file
st.markdown("### 📁 Upload Investors File")
investor_file = st.file_uploader("Upload `investors.xlsx` file", type=["xlsx"])
if not investor_file:
    st.warning("⚠️ Please upload an `investors.xlsx` file to proceed.")
    st.stop()

@st.cache_data
def load_investors(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df = df.dropna(subset=["Linkedin Info"])
    return df

investor_df = load_investors(investor_file)

# Pitch input section
st.markdown("---")
st.header("📝 Paste Your Pitch")

text_input = st.text_area("Paste your pitch below 👇", height=200)
submit_button = st.button("🎯 Submit Pitch")

if submit_button:
    pitch_text = text_input.strip()
    
    if not pitch_text:
        st.warning("⚠️ Please provide some pitch text before submitting.")
        st.stop()

    @st.cache_resource
    def get_vectorizer_and_matrix(texts):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix

    vectorizer, investor_matrix = get_vectorizer_and_matrix(investor_df["Linkedin Info"])
    pitch_vector = vectorizer.transform([pitch_text])

    cosine_sim = cosine_similarity(pitch_vector, investor_matrix).flatten()
    investor_df["Similarity"] = cosine_sim
    top_matches = investor_df.sort_values("Similarity", ascending=False).head(3)

    st.markdown("---")
    st.subheader("🏆 Top 3 Investor Matches")
    for idx, row in top_matches.iterrows():
        st.markdown(f"""
        🔹 **Name:** {row['Name']}  
        📧 **Email:** {row['Email']}  
        🔗 **LinkedIn:** [{row['Linkedin URL']}]({row['Linkedin URL']})  
        📊 **Similarity Score:** `{row['Similarity']:.4f}`  
        ---  
        """)

    st.success("✅ Done! Ready to reach out? Start those conversations 🚀")
