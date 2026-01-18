🍽️ Swiggy Restaurant Recommendation System
📌 Project Overview

This project builds a Restaurant Recommendation System using Swiggy’s restaurant data. The system suggests similar restaurants based on user preferences such as cuisine, location, ratings, and pricing.
An interactive Streamlit web application is developed to make recommendations easily accessible to users.

🎯 Objectives

Recommend restaurants similar to a selected restaurant or user preference

Handle large categorical data efficiently

Provide fast and user-friendly recommendations through a web interface

🧠 Techniques & Concepts Used
🔹 Data Preprocessing

Handling missing and inconsistent values

Cleaning location and city information

Selecting relevant features for recommendation

🔹 One-Hot Encoding

Converted categorical features (cuisine, city, restaurant type, etc.) into numerical format

Used pandas.get_dummies() for efficient encoding

🔹 Clustering & Similarity Methods

K-Means Clustering: Groups similar restaurants based on features

Enables fast and relevant restaurant suggestions

🔹 Streamlit Application Development

Interactive UI for selecting restaurants or filters

Displays top recommended restaurants instantly

Lightweight and easy-to-use web interface

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn

Web Framework: Streamlit

Machine Learning: K-Means

🚀 How to Run the Project

Clone the repository

git clone <repository-url>


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run clientEnv/Scripts/app.py

📊 Output

Restaurant recommendations based on similarity

Cluster-based restaurant grouping

Interactive recommendation dashboard

📈 Future Enhancements

Add user ratings and reviews for better personalization

Use collaborative filtering

Deploy the app on cloud platforms

Optimize performance for very large datasets
