import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\IMDB_Movie_Recommendation_project\src\cleaned_movies.csv")
    df = df.dropna(subset=["Movie Name", "Storyline"])
    return df

df = load_data()

# -----------------------
# MODEL
# -----------------------
@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["Storyline"])
    return tfidf, cosine_similarity(matrix)

tfidf, similarity = build_model(df)

# -----------------------
# RECOMMENDER
# -----------------------
def recommend(index, n):
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i[0] for i in scores]]

# -----------------------
# UI
# -----------------------
st.title("🎬 IMDB Movie Recommendation System")
st.write("Enter a storyline and get recommendations")

storyline = st.text_area("Enter Storyline")
num = st.slider("Number of Recommendations", 1, 10, 5)

# -----------------------
# RUN APP
# -----------------------
if st.button("Recommend"):

    if not storyline.strip():
        st.warning("Please enter a storyline")
    else:

        temp = tfidf.transform(df["Storyline"].tolist() + [storyline])
        sim_input = cosine_similarity(temp[-1], temp[:-1])

        best_match = sim_input.argmax()
        results = recommend(best_match, num)

        st.subheader("🎯 Top Recommendations")

        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                st.markdown(f"**{row['Movie Name']}**")
                st.caption(row["Storyline"][:80] + "...")