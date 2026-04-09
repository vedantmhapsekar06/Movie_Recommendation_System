# =============================================================================
# MOVIE RECOMMENDATION SYSTEM — Content-Based Filtering
# Updated for movies_cleaned.csv
# =============================================================================

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. LOAD DATA
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    print("\n" + "="*55)
    print(" MOVIE RECOMMENDER (Cleaned Dataset)")
    print("="*55)
    print(f"\nMovies loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    return df


# 2. PREPROCESS
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title", "genres"])

    # Remove useless rows
    df = df[df["genres"] != "(no genres listed)"].copy()

    # Clean genres
    df["genre_tokens"] = (
        df["genres"]
        .str.replace("-", "", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()   # normalize
    )

    df["title"] = df["title"].str.strip()

    # Remove duplicate titles (keep first)
    df = df.drop_duplicates(subset="title")

    df = df.reset_index(drop=True)

    print(f"After cleaning: {len(df)} movies\n")
    return df


# 3. TF-IDF
def build_tfidf(df):
    tfidf = TfidfVectorizer(
        stop_words=None,
        token_pattern=r"[^\s]+"
    )

    matrix = tfidf.fit_transform(df["genre_tokens"])

    print(f"TF-IDF shape: {matrix.shape}\n")
    return matrix


# 4. SIMILARITY
def build_similarity(matrix):
    print("Computing similarity...")
    sim = cosine_similarity(matrix)
    print("Done!\n")
    return sim


# 5. RECOMMENDATION
def recommend(movie_name, df, sim_matrix, top_n=5):
    movie_name = movie_name.lower().strip()

    # Try exact match
    matches = df[df["title"].str.lower() == movie_name]

    # If not found → partial match
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(movie_name)]

    if matches.empty:
        print("Movie not found!\n")
        return None

    idx = matches.index[0]

    print(f"\nSelected: {df.loc[idx, 'title']}")
    print(f"Genres : {df.loc[idx, 'genres']}\n")

    scores = list(enumerate(sim_matrix[idx]))

    # Sort
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores[1:top_n+1]:
        results.append({
            "title": df.loc[i, "title"],
            "genres": df.loc[i, "genres"],
            "score": round(score, 4)
        })

    return pd.DataFrame(results)


# 6. DISPLAY
def show(results):
    if results is None:
        return

    print("\nTop Recommendations:\n")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['title']}")
        print(f"   Genres: {row['genres']}")
        print(f"   Score : {row['score']}\n")


# 7. MAIN
def main():
    path = "movies_cleaned.csv"   # <-- your file

    df = load_data(path)
    df = preprocess(df)

    tfidf_matrix = build_tfidf(df)
    sim_matrix = build_similarity(tfidf_matrix)

    while True:
        name = input("Enter movie (or 'exit'): ")

        if name.lower() in ["exit", "quit"]:
            break

        results = recommend(name, df, sim_matrix)
        show(results)


if __name__ == "__main__":
    main()