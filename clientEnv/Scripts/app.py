import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    page_icon="C:/MAMP/htdocs/SwiggyRestaurantRecommendations/clientEnv/Scripts/images.jfif",
    layout="wide"
)

swiggy_df = pd.read_csv(r"C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\swiggy.csv")

# swiggy_df.head()

swiggy_df = swiggy_df.drop(['lic_no', 'link', 'address', 'menu'], axis=1)

# swiggy_df.size
# swiggy_df.shape
# swiggy_df.info()

swiggy_df['rating'] = swiggy_df['rating'].replace('--', np.nan).astype(float)
swiggy_df['cost'] = pd.to_numeric(
    swiggy_df['cost'].str.replace('₹', '', regex=False),
    errors='coerce'
)
swiggy_df['rating_count'] = (
    swiggy_df['rating_count']
    .str.extract('(\d+)')   # extract digits only
    .astype(float)
)

# swiggy_df.describe()

# swiggy_df.select_dtypes(include='number').skew()

swiggy_df['rating_count_log'] = np.log1p(swiggy_df['rating_count'])
swiggy_df['cost_log'] = np.log1p(swiggy_df['cost'])

swiggy_df['rating'].fillna(swiggy_df['rating'].median(), inplace=True)
swiggy_df['rating_count'].fillna(swiggy_df['rating_count'].median(), inplace=True)
swiggy_df['cost'].fillna(swiggy_df['cost'].median(), inplace=True)

swiggy_df['rating_count_log'].fillna(swiggy_df['rating_count_log'].median(), inplace=True)
swiggy_df['cost_log'].fillna(swiggy_df['cost_log'].median(), inplace=True)

swiggy_df['city_name'] = (
    swiggy_df['city']
    .str.extract(r',\s*(.*)$', expand=False)   # FIXED regex
    .fillna(swiggy_df['city'])
    .str.strip()
)

cat_cols = ['name', 'cuisine']
swiggy_df[cat_cols] = swiggy_df[cat_cols].fillna('Unknown')

# swiggy_df.isnull().sum()
# swiggy_df.duplicated().sum()

swiggy_df.to_csv("cleaned_data.csv", index=False)

swiggy_cleaned_df = pd.read_csv(r'C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\cleaned_data.csv')

cols = ['city_name', 'cuisine']

swiggy_cleaned_df['cuisine'] = swiggy_cleaned_df['cuisine'].str.lower().str.strip()
swiggy_cleaned_df['cuisine'] = swiggy_cleaned_df['cuisine'].str.split(',')
swiggy_cleaned_df['cuisine'] = swiggy_cleaned_df['cuisine'].apply(lambda x: [i.strip() for i in x])

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
cuisine_encoded = mlb.fit_transform(swiggy_cleaned_df['cuisine'])

cuisine_df = pd.DataFrame(
    cuisine_encoded,
    columns=mlb.classes_,
    index=swiggy_cleaned_df.index
)

city_encoded = pd.get_dummies(swiggy_cleaned_df['city_name'], prefix='city', drop_first=True)

cuisine_df.index = swiggy_cleaned_df.index
city_encoded.index = swiggy_cleaned_df.index

swiggy_df_model = pd.concat(
    [
        swiggy_cleaned_df[['rating','rating_count_log','cost_log']],
        city_encoded,
        cuisine_df
    ],
    axis=1
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
swiggy_df_model[['rating','rating_count_log','cost_log']] = scaler.fit_transform(
    swiggy_df_model[['rating','rating_count_log','cost_log']]
)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
swiggy_cleaned_df['cluster'] = kmeans.fit_predict(swiggy_df_model)

swiggy_df_model.to_pickle("encoded_data.pkl")
swiggy_df_model.to_csv("encoded_data.csv")

cleaned = pd.read_csv(r'C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\cleaned_data.csv')
encoded = pd.read_pickle(r'C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\encoded_data.pkl')

# cleaned.index.equals(encoded.index)

cleaned['row_id'] = cleaned.index
encoded['row_id'] = encoded.index

user_vec = pd.DataFrame(columns=swiggy_df_model.columns)
user_vec.loc[0] = 0



col1, col2, col3 = st.columns([1, 12, 1])
with col2:
    st.markdown("<h2 style='text-align: center;'>Swiggy Restaurant Recommendation System</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 5, 3])
with col2:
    form = st.form(key="login_form")

    userCity = form.text_input("City")
    userCuisine = form.text_input("Cuisine")
    userRating = form.text_input("Rating")
    userPrice = form.text_input("Price")

    submitButton = form.form_submit_button("Submit")
    
    if submitButton:
            if userCity == '':
                st.error('Enter the City')
            elif userCuisine == '':
                st.error('Enter the Cuisine')
            elif userRating == '':
                st.error('Enter the Rating')
            elif userPrice == '':
                st.error('Enter the Price')
            else:
                user_vec = pd.DataFrame(columns=swiggy_df_model.columns)
                user_vec.loc[0] = 0

                user_city = userCity
                user_cuisines = userCuisine
                user_rating = float(userRating)
                user_budget = float(userPrice)
                
                user_vec['rating'] = user_rating
                user_vec['rating_count_log'] = 0
                user_vec['cost_log'] = np.log1p(user_budget)

                city_col = "city_" + user_city
                if city_col in user_vec.columns:
                    user_vec[city_col] = 1

                for cuisine in user_cuisines:
                    if cuisine in user_vec.columns:
                        user_vec[cuisine] = 1
                
                user_vec[['rating','rating_count_log','cost_log']] = scaler.transform(
                    user_vec[['rating','rating_count_log','cost_log']]
                )
                
                # Predict the cluster
                user_cluster = kmeans.predict(user_vec)[0]
                
                recommendations = swiggy_cleaned_df[swiggy_cleaned_df['cluster']==user_cluster]
                
                recommendations = recommendations.sort_values(
                    by='rating', ascending=False
                )

                recommendations[['name','city','cuisine','rating','cost']].head(10)


###### App CSS - Start ######
st.markdown("""
<style>
    /* Adjust this selector based on what F12 shows */
    .e1mlolmg0st-emotion-cache-8atqhb { 
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
    }

    .st-emotion-cache-1tvzk6f {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Targets the primary container that wraps all your content */
    .block-container {
        padding-top: 2.5rem; /* Reduces the top padding from the default large value */
        padding-bottom: 0rem; /* Reduces default bottom padding */
        padding-left: 0rem;
        /* You can adjust the left/right padding if needed, but often not required */
    }
    
    /* Targets the header/toolbar placeholder element (often invisible but reserves space) */
    .stApp > header {
        display: none; 
    }
    
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Targets the primary sidebar container */
    [data-testid="stSidebar"] {
        background-color: #0A2885;
        color: #FFFFFF
    }
    div.stButton > button:first-child {
        background-color: #415eb9;
        color: #FFFFFF
    }
    # div.stButton > button:first-child:hover {
    #     background-color: #415eb9;
    # }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Base style for the navigation text (makes it look like a link) */
    .nav-link-text {
        padding: 8px 15px;
        margin: 5px 0;
        cursor: pointer;
        transition: background-color 0.2s ease, color 0.2s ease;
        border-radius: 4px;
        font-size: 16px;
        font-weight: 500;
        color: #333333; 
        display: block; 
    }

    /* Style on mouse hover (The highlight effect) */
    .nav-link-text:hover {
        background-color: #e0f2ff; /* Light blue highlight on hover */
        color: #007bff; /* Blue text color on hover */
    }

    /* Style for the currently active/selected page */
    .nav-link-text.active {
        background-color: #007bff !important; /* Blue background */
        color: white !important; /* White text */
        font-weight: 600;
    }

    /* --- STREAMLIT BUTTON OVERRIDES (Makes the button look like text) --- */

    /* Targets ALL buttons inside the sidebar for custom styling */
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        border: none;
        text-align: left;
        width: 100%;
        /* Apply the base nav-link style to the button structure */
        padding: 8px 15px; 
        margin-top: -10px; /* Reduces extra space above the button */
        font-size: 16px;
        font-weight: 500;
    }

    /* Reset hover for standard buttons, and let the custom CSS handle it */
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: transparent; 
    }

    </style>
""", unsafe_allow_html=True)
###### App CSS - End ######