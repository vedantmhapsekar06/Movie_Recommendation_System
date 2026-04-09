**Movie Recommendation System**
A content-based movie recommender built with Python, Flask, and a clean web dashboard. Enter any movie title and instantly get similar movie suggestions powered by TF-IDF vectorization and cosine similarity.

** Features**

 Smart autocomplete — search from thousands of movie titles in real time
 Content-based filtering — recommendations based on genre similarity using TF-IDF + cosine similarity
 Interactive dashboard — visualizes top genres, similarity scores, and dataset stats
 REST API — clean Flask endpoints for movies, recommendations, and stats
 CLI mode — run the recommender directly from the terminal without a browser


** Project Structure**
movie-recommendation-system/
│
├── app.py                  # Flask backend + API routes
├── movie_recommender.py    # Standalone CLI recommender
├── dashboard.html          # Frontend dashboard (served by Flask)
└── movies_cleaned.csv      # Cleaned movie dataset
└── requirements.txt        # requirements needed

 **Getting Started**
1. Clone the repository
bashgit clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
2. Install dependencies
bashpip install flask flask-cors pandas scikit-learn
3. Run the web app
bashpython app.py
Then open your browser at: http://localhost:5000
The dashboard will load automatically.

** CLI Mode**
Prefer the terminal? Run the standalone recommender:
bashpython movie_recommender.py
Enter movie (or 'exit'): Toy Story

Selected: Toy Story
Genres : Adventure|Animation|Children|Comedy|Fantasy

Top Recommendations:
1. Antz
   Genres: Adventure|Animation|Children|Comedy|Fantasy
   Score : 1.0

2. Monsters, Inc.
   Genres: Adventure|Animation|Children|Comedy|Fantasy
   Score : 1.0
...

** API Endpoints**
MethodEndpointDescriptionGET/Serves the dashboardGET/api/moviesReturns all movie titles and genresGET/api/recommend?movie=<title>&n=<count>Returns top N similar moviesGET/api/statsReturns total movie count and top genres
Example
GET /api/recommend?movie=Toy Story&n=5
json{
  "selected": { "title": "Toy Story", "genres": "Adventure|Animation|Children|Comedy|Fantasy" },
  "recommendations": [
    { "title": "Antz", "genres": "Adventure|Animation|Children|Comedy|Fantasy", "score": 1.0 },
    ...
  ]
}

** How It Works**

Load — reads movies_cleaned.csv and drops rows with missing/invalid genres
Preprocess — cleans genre strings and tokenizes them (e.g. "Action|Sci-Fi" → "action scifi")
TF-IDF — converts genre tokens into numerical vectors
Cosine Similarity — builds a similarity matrix across all movies
Recommend — for a queried movie, returns the top N closest matches by similarity score


A score of 1.0 means the genres are identical. A score of 0.0 means no shared genres.


** Tech Stack**
LayerTechnologyBackendPython, Flask, Flask-CORSMLscikit-learn (TF-IDF, Cosine Similarity)DatapandasFrontendHTML, CSS, Vanilla JavaScript

** Dataset**
The project uses movies_cleaned.csv — a pre-cleaned movie dataset with title and genres columns. Genres are pipe-separated (e.g. Action|Comedy|Drama).

You can swap in your own dataset as long as it has title and genres columns.


