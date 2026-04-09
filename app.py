from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, webbrowser, threading

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load & preprocess ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  Movie Recommendation System")
print("="*50)
print("\n⏳ Loading and processing data...")

df = pd.read_csv("movies_cleaned.csv")
df = df.dropna(subset=["title", "genres"])
df = df[df["genres"] != "(no genres listed)"].copy()
df["genre_tokens"] = (
    df["genres"]
    .str.replace("-", "", regex=False)
    .str.replace("|", " ", regex=False)
    .str.lower()
)
df["title"] = df["title"].str.strip()
df = df.drop_duplicates(subset="title").reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words=None, token_pattern=r"[^\s]+")
matrix = tfidf.fit_transform(df["genre_tokens"])
sim_matrix = cosine_similarity(matrix)

print(f"✅ {len(df)} movies loaded and similarity matrix built.")


# ── Serve dashboard HTML ──────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return send_from_directory(".", "dashboard.html")


# ── API Routes ────────────────────────────────────────────────────────────────
@app.route("/api/movies")
def get_movies():
    return jsonify(df[["title", "genres"]].to_dict("records"))


@app.route("/api/recommend")
def recommend():
    movie_name = request.args.get("movie", "").lower().strip()
    top_n      = int(request.args.get("n", 5))

    matches = df[df["title"].str.lower() == movie_name]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(movie_name, na=False)]
    if matches.empty:
        return jsonify({"error": "Movie not found"}), 404

    idx      = matches.index[0]
    selected = {"title": df.loc[idx, "title"], "genres": df.loc[idx, "genres"]}

    scores  = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    results = [
        {
            "title":  df.loc[i, "title"],
            "genres": df.loc[i, "genres"],
            "score":  round(float(score), 4),
        }
        for i, score in scores[1: top_n + 1]
    ]

    return jsonify({"selected": selected, "recommendations": results})


@app.route("/api/stats")
def stats():
    genre_counts = df["genres"].str.split("|").explode().value_counts().head(10)
    return jsonify({
        "total_movies": int(len(df)),
        "genres": [{"name": k, "count": int(v)} for k, v in genre_counts.items()],
    })


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    url = "http://localhost:5000"
    print("\n" + "="*50)
    print(f"  🚀 Dashboard running!")
    print(f"  👉  Open this link: {url}")
    print("="*50)
    print("\n  Press CTRL+C to stop the server.\n")

    # Auto-open browser after 1 second
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(port=5000, debug=False)
