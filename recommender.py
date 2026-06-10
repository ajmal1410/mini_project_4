import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.df["Cleaned_Storyline"] = self.df["Cleaned_Storyline"].fillna("")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["Cleaned_Storyline"]
        )

    def recommend_movies(self, storyline, top_n=5):

        user_vector = self.vectorizer.transform([storyline])

        scores = cosine_similarity(
            user_vector,
            self.tfidf_matrix
        ).flatten()

        top_indices = scores.argsort()[-top_n:][::-1]

        return self.df.iloc[top_indices]