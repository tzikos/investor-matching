# Pitch Deck to Investor Brief Matcher with Simple Streamlit UI

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-large-en")

model = load_model()

# Streamlit app UI
st.title("Startup Pitch Matcher")
st.markdown("Match your startup pitch to an investor brief using semantic similarity.")

with st.form("pitch_form"):
    st.header("Pitch Deck")
    pitch_problem = st.text_area("Problem", "Managing remote teams across time zones creates communication delays and productivity loss.")
    pitch_solution = st.text_area("Solution", "A smart async communication platform with timezone-aware task planning and voice memos.")
    pitch_market = st.text_area("Market", "Targeting tech startups and distributed teams with over $10B market potential.")
    pitch_traction = st.text_area("Traction", "Over 15k users and 200 paying teams in 6 months.")
    pitch_team = st.text_area("Team", "Founders from Google and Atlassian with 10+ years in SaaS.")

    st.header("Investor Brief")
    investor_sectors = st.text_area("Sectors", "Future of work, productivity tools, SaaS.")
    investor_stage = st.text_area("Stage", "Seed to Series A.")
    investor_geography = st.text_area("Geography", "US and Europe.")
    investor_themes = st.text_area("Themes", "Remote work infrastructure, B2B SaaS, collaboration.")

    submitted = st.form_submit_button("Match Now")

if submitted:
    pitch = {
        "problem": pitch_problem,
        "solution": pitch_solution,
        "market": pitch_market,
        "traction": pitch_traction,
        "team": pitch_team
    }

    investor = {
        "sectors": investor_sectors,
        "stage": investor_stage,
        "geography": investor_geography,
        "themes": investor_themes
    }

    def embed_sections(sections):
        return {k: model.encode(v, convert_to_tensor=True) for k, v in sections.items()}

    pitch_embeddings = embed_sections(pitch)
    investor_embeddings = embed_sections(investor)

    weights = {
        "sectors": 1.0,
        "themes": 1.0,
        "stage": 0.5,
        "geography": 0.3
    }

    similarities = []
    for pk, p_emb in pitch_embeddings.items():
        for ik, i_emb in investor_embeddings.items():
            sim = util.cos_sim(p_emb, i_emb).item()
            weight = weights.get(ik, 0.2)
            similarities.append((pk, ik, sim, weight))

    weighted_sum = sum(sim * w for _, _, sim, w in similarities)
    weight_total = sum(w for *_, w in similarities)
    final_score = weighted_sum / weight_total

    st.subheader("Match Score")
    st.metric(label="Semantic Match Score", value=f"{final_score:.4f}")

    st.subheader("Section Similarities")
    for pk, ik, sim, w in similarities:
        st.write(f"**{pk}** ↔ **{ik}** | Similarity: `{sim:.4f}` | Weight: `{w}`")
