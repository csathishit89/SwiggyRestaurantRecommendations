# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

if "results" not in st.session_state:
    st.session_state.results = None
    
st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    layout="wide"
)

ARTIFACT_DIR = "artifacts"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    cleaned = pd.read_pickle(f"{ARTIFACT_DIR}/cleaned.pkl")
    encoded = pd.read_pickle(f"{ARTIFACT_DIR}/encoded.pkl")
    scaler = pickle.load(open(f"{ARTIFACT_DIR}/scaler.pkl", "rb"))
    kmeans = pickle.load(open(f"{ARTIFACT_DIR}/kmeans.pkl", "rb"))
    return cleaned, encoded, scaler, kmeans

with st.spinner("Loading recommendation engine..."):
    cleaned_df, encoded_df, scaler, kmeans = load_artifacts()

# ---------------- UI ----------------
st.markdown(
    "<h2 style='text-align:center;'>Swiggy Restaurant Recommendation System</h2>",
    unsafe_allow_html=True
)

def clear_results():
    st.session_state.results = None

with st.form("recommendation_form"):
    city = st.text_input("City")
    cuisine = st.text_input("Cuisine (comma separated)")
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    price = st.number_input("Budget (₹)", min_value=50)

    submitted = st.form_submit_button("Get Recommendations")

# ---------------- PREDICTION ----------------
if submitted:
    user_vec = pd.DataFrame(
        0,
        index=[0],
        columns=encoded_df.columns
    )

    user_vec['rating'] = rating
    user_vec['rating_count_log'] = 0
    user_vec['cost_log'] = np.log1p(price)

    city_col = f"city_{city}"
    if city_col in user_vec.columns:
        user_vec[city_col] = 1

    cuisines = [c.strip().lower() for c in cuisine.split(",")]
    for c in cuisines:
        if c in user_vec.columns:
            user_vec[c] = 1

    user_vec[['rating','rating_count_log','cost_log']] = scaler.transform(
        user_vec[['rating','rating_count_log','cost_log']]
    )

    cluster = kmeans.predict(user_vec)[0]

    results = (
        cleaned_df[cleaned_df['cluster'] == cluster]
        .sort_values(by="rating", ascending=False)
        .head(10)
    )

    st.session_state.results = results
    
    if st.session_state.results is not None:
        st.subheader("🍽 Recommended Restaurants")
        st.dataframe(
            results[['name', 'city', 'cuisine', 'rating', 'cost']],
            use_container_width=True
        )