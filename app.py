from flask import Flask, render_template, jsonify, request
from testingcode import NewsRecommender
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Recommender (Global instance to avoid reloading SBERT model on every request)
print("Initializing Recommender System...")
recommender = NewsRecommender()
print("Recommender Initialized.")

# Default static user data (copied from testingcode.py for demo purposes)
DEFAULT_USER_DATA = {
    "categories": ["Technology", "Sports"],
    "likes": [
        {"text": "Latest Marvel movie breaks box office records with huge opening weekend.", "timestamp": datetime.now()},
        {"text": "Review of the new Christopher Nolan film: A masterpiece of cinema.", "timestamp": datetime.now()},
        {"text": "Grammy awards winners announced: Taylor Swift takes home album of the year.", "timestamp": datetime.now()}
    ],
    "history": [
        {"text": "NBA Finals: Lakers secure victory in thrilling overtime game.", "timestamp": datetime.now()},
        {"text": "Manchester United transfer rumors: Top striker looking to move.", "timestamp": datetime.now()},
        {"text": "Netflix announces new season of hit fantasy series.", "timestamp": datetime.now()},
        {"text": "Formula 1 race results: Verstappen wins again.", "timestamp": datetime.now()}
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        # Get user data from request if provided, else use default
        user_input = request.json
        
        # Prepare User Data
        # Start with default
        current_data = DEFAULT_USER_DATA.copy()
        
        # 1. Override Categories
        if user_input and 'categories' in user_input:
             current_data['categories'] = user_input['categories']
             
        # 2. Override Likes (Simulation)
        if user_input and 'custom_likes' in user_input and user_input['custom_likes']:
            # Split lines and create dict structure
            lines = [l.strip() for l in user_input['custom_likes'].split('\n') if l.strip()]
            current_data['likes'] = [{"text": l, "timestamp": datetime.now()} for l in lines]
            
        # 3. Override History (Simulation)
        if user_input and 'custom_history' in user_input and user_input['custom_history']:
            lines = [l.strip() for l in user_input['custom_history'].split('\n') if l.strip()]
            current_data['history'] = [{"text": l, "timestamp": datetime.now()} for l in lines]

        recommender.user_data = current_data
        
        # Generate embedding
        recommender.generate_user_embedding()
        
        # Get recommendations (Static Only)
        articles = recommender.recommend_articles(num_recommendations=20)
        
        # Format for JSON response
        formatted_articles = []
        for i, art in enumerate(articles, 1):
            formatted_articles.append({
                "id": i,
                "title": art.get("title", "No title"),
                "body": art.get("body", "")[:200] + "...", # Snippet
                "category": art.get("category", "General"),
                "score": round(art.get("recommendation_score", 0), 2),
                "date": art.get("date", str(datetime.now().date())), # Fallback or use art['dateTime']
                "url": art.get("url", "#") # Assuming 'url' might be in the raw article object
            })
            
        return jsonify({"status": "success", "articles": formatted_articles})

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
