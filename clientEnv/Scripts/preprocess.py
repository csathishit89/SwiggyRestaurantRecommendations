# preprocess.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv( r"C:\MAMP\htdocs\SwiggyRestaurantRecommendations\clientEnv\Scripts\swiggy.csv")

df = df.drop(['lic_no', 'link', 'address', 'menu'], axis=1)

# ---------------- CLEAN ----------------
df['rating'] = df['rating'].replace('--', np.nan).astype(float)

df['cost'] = pd.to_numeric(
    df['cost'].str.replace('₹', '', regex=False),
    errors='coerce'
)

df['rating_count'] = (
    df['rating_count']
    .str.extract(r'(\d+)')
    .astype(float)
)

df['rating'].fillna(df['rating'].median(), inplace=True)
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
df['cost'].fillna(df['cost'].median(), inplace=True)

df['rating_count_log'] = np.log1p(df['rating_count'])
df['cost_log'] = np.log1p(df['cost'])

df['city_name'] = (
    df['city']
    .str.extract(r',\s*(.*)$', expand=False)
    .fillna(df['city'])
    .str.strip()
)

df[['name', 'cuisine']] = df[['name', 'cuisine']].fillna("Unknown")

# ---------------- CUISINE ENCODING ----------------
df['cuisine'] = (
    df['cuisine']
    .str.lower()
    .str.split(',')
    .apply(lambda x: [i.strip() for i in x])
)

mlb = MultiLabelBinarizer()
cuisine_encoded = mlb.fit_transform(df['cuisine'])
cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_)

# ---------------- CITY ENCODING ----------------
city_df = pd.get_dummies(df['city_name'], prefix="city", drop_first=True)

# ---------------- FINAL MODEL DF ----------------
model_df = pd.concat(
    [
        df[['rating', 'rating_count_log', 'cost_log']],
        city_df,
        cuisine_df
    ],
    axis=1
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
model_df[['rating', 'rating_count_log', 'cost_log']] = scaler.fit_transform(
    model_df[['rating', 'rating_count_log', 'cost_log']]
)

# ---------------- KMEANS ----------------
kmeans = KMeans(n_clusters=6, n_init=10)
df['cluster'] = kmeans.fit_predict(model_df)

# ---------------- SAVE ARTIFACTS ----------------
df.to_pickle(f"{ARTIFACT_DIR}/cleaned.pkl")
model_df.to_pickle(f"{ARTIFACT_DIR}/encoded.pkl")

pickle.dump(scaler, open(f"{ARTIFACT_DIR}/scaler.pkl", "wb"))
pickle.dump(kmeans, open(f"{ARTIFACT_DIR}/kmeans.pkl", "wb"))
pickle.dump(mlb, open(f"{ARTIFACT_DIR}/mlb.pkl", "wb"))

print("✅ Preprocessing completed successfully")