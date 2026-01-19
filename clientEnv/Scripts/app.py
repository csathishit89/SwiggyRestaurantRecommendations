# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

if "results" not in st.session_state:
    st.session_state.results = None
    
st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    page_icon="C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\swiggy-favicon.png",
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
    userCity = st.text_input("City")
    userCuisine = st.text_input("Cuisine (comma separated)")
    userRating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    userPrice = st.number_input("Budget (₹)", min_value=0)

    submitted = st.form_submit_button("Get Recommendations")

    if submitted:
            if userCity == '':
                st.error('Enter the City')
            elif userCuisine == '':
                st.error('Enter the Cuisine')
            elif userRating == '':
                st.error('Enter the Rating')
            elif userPrice == '':
                st.error('Enter the Price')
            else:
                # ---------------- PREDICTION ----------------
                # ---------- Build user vector ----------
                user_vec = pd.DataFrame(0, index=[0], columns=encoded_df.columns)

                user_vec['rating'] = userRating
                user_vec['rating_count_log'] = 0
                user_vec['cost_log'] = np.log1p(userPrice)

                # City
                city_clean = userCity.strip().lower()
                city_col = f"city_{city_clean}"
                if city_col in user_vec.columns:
                    user_vec[city_col] = 1
                
                # Cuisine
                cuisines = [c.strip().lower() for c in userCuisine.split(",")]
                for c in cuisines:
                    if c in user_vec.columns:
                        user_vec[c] = 1

                # ---------- Scale ----------
                user_vec[['rating','rating_count_log','cost_log']] = scaler.transform(
                    user_vec[['rating','rating_count_log','cost_log']]
                )

                # ---------- KMeans TRANSFORM ----------
                distances_to_centers = kmeans.transform(user_vec)
                user_cluster = distances_to_centers.argmin(axis=1)[0]

                # ---------- Filter cluster ----------
                cluster_data = cleaned_df[
                    cleaned_df['cluster'] == user_cluster
                ].copy()

                cluster_encoded = encoded_df.loc[cluster_data.index]

                cluster_matrix = cluster_encoded.astype(float).values
                user_matrix = user_vec.astype(float).values


                # ---------- Euclidean distance to user ----------
                cluster_data['distance'] = np.linalg.norm(
                    cluster_matrix - user_matrix,
                    axis=1
                )

                # ---------- Final recommendations ----------
                recommendations = (
                    cluster_data
                    .sort_values(by=['distance','rating'], ascending=[False,False])
                    .head(10)
                )

                st.session_state.results = recommendations
                
                if st.session_state.results is not None:
                    st.subheader("🍽 Recommended Restaurants")
                    st.dataframe(
                        recommendations[['name', 'city', 'cuisine', 'rating', 'cost']],
                        use_container_width=True
                    )