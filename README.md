# AI News Recommender

An intelligent news recommendation system using **SBERT (Sentence-BERT)** for semantic similarity and **Maximal Marginal Relevance (MMR)** for diverse ranking.

## 🚀 Features

-   **Semantic Matching**: Uses `all-MiniLM-L6-v2` to understand the *meaning* of news articles, not just keywords.
-   **Simulation Mode**: A fully offline, static testing environment to verify recommendation logic without external API limits.
-   **Hybrid Filtering**: Combines user interests (Category), explicit likes (Content-based), and reading history (Time-decayed behavior).
-   **Interactive UI**: A clean, modern web interface to simulate different user personas and see real-time recommendation updates.

## 🛠️ Tech Stack

-   **Backend**: Python, Flask
-   **ML/NLP**: Sentence-Transformers (SBERT), Scikit-Learn (Cosine Similarity)
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **Data**: Static Article Pool (Offline Simulation)

## 📦 Installation & Setup

1.  **Install Dependencies**:
    ```bash
    pip install flask sentence-transformers scikit-learn numpy python-dotenv
    ```

2.  **Run the Application**:
    Double-click `run_app.bat` 
    *OR* run via terminal:
    ```bash
    python app.py
    ```

3.  **Access**:
    Open your browser to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## 🎮 How to Use (Simulation Mode)

This app is currently configured in **Static Simulation Mode** for testing purposes.

1.  **Select Active Interest**: Choose a category (e.g., "Technology") to filter the "Active Interest" weight.
2.  **Simulate Likes**: In the "Liked Articles" box, type descriptions of content the user likes (one per line).
    *   *Example:* "I love rockets and space exploration."
3.  **Simulate History**: In the "Reading History" box, type content the user has read recently.
4.  **Get Recommendations**: Click the button to see how the SBERT model re-ranks the static article pool based on your inputs.

## 📂 Project Structure

-   `app.py`: Flask server handling API requests.
-   `testingcode.py`: Core logic for the Recommender System (SBERT, MMR, Static Data).
-   `templates/index.html`: Frontend user interface.
-   `run_app.bat`: Quick-start script for Windows.
