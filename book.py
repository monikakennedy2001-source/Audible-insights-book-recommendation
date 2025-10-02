import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set(style="whitegrid")

st.set_page_config(page_title="Audible Book Recommendation", layout="wide")

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_parquet("cleaned_clustered_books.parquet")
    X_norm = sparse.load_npz("features_norm.npz")
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("mlb_genres.pkl", "rb") as f:
        mlb = pickle.load(f)
    return df, X_norm, tfidf, mlb

df, X_norm, tfidf, mlb = load_data()

def fuzzy_match_title(title, titles, cutoff=0.6):
    matches = get_close_matches(title, titles, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

def content_based_recommend(book_title, df, features, top_n=5):
    matched_title = fuzzy_match_title(book_title, df['Book Name'].tolist())
    if not matched_title:
        return f"Book '{book_title}' not found or no close match."
    
    idx = df.index[df['Book Name'] == matched_title][0]
    book_vec = features[idx]
    sims = cosine_similarity(book_vec, features).flatten()
    sims[idx] = -1  # exclude same book
    top_indices = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Book Name', 'Author', 'Rating']]

st.title("📚 Audible Book Recommendation System with EDA")

# Sidebar - EDA selector
st.sidebar.header("Explore Dataset Insights")
eda_option = st.sidebar.selectbox("Select an EDA visualization:", 
                                  ["None", 
                                   "Rating Distribution", 
                                   "Number of Reviews Distribution (log scale)", 
                                   "Price Distribution", 
                                   "Top Genres", 
                                   "Feature Correlation", 
                                   "Listening Time vs Rating"])

if eda_option != "None":
    st.header(f"EDA: {eda_option}")
    
    if eda_option == "Rating Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Rating"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Number of Reviews Distribution (log scale)":
        fig, ax = plt.subplots()
        sns.histplot(df["Number of Reviews"].apply(lambda x: x+1), bins=50, log_scale=True, ax=ax)
        ax.set_xlabel("Number of Reviews (log scale)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Price Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Price"], bins=30, ax=ax)
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Top Genres":
        all_genres = [genre for sublist in df["Genres"] for genre in sublist]
        genre_counts = Counter(all_genres)
        top_genres = genre_counts.most_common(20)
        genres, counts = zip(*top_genres)
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=list(counts), y=list(genres), ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Genre")
        st.pyplot(fig)

    elif eda_option == "Feature Correlation":
        fig, ax = plt.subplots(figsize=(8,6))
        corr = df[["Rating", "Number of Reviews", "Price", "Listening Time Minutes"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif eda_option == "Listening Time vs Rating":
        fig, ax = plt.subplots()
        sns.scatterplot(x="Listening Time Minutes", y="Rating", data=df, alpha=0.5, ax=ax)
        ax.set_xlabel("Listening Time (Minutes)")
        ax.set_ylabel("Rating")
        st.pyplot(fig)

st.header("Book Recommendation")

book_input = st.text_input("Enter a book title you like:")

if st.button("Get Recommendations"):
    if book_input:
        recs = content_based_recommend(book_input, df, X_norm, top_n=5)
        if isinstance(recs, str):
            st.error(recs)
        else:
            st.write("Top recommendations based on your input:")
            st.dataframe(recs)
    else:
        st.warning("Please enter a book title.")
